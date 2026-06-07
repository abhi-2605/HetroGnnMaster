"""Diagnostic: run results/checkpoints/best_model.pt over the VAL split and
identify exactly which threats (label 0=ransom / 2=malware) get predicted Benign
(label 1) -- the "Threats missed" count -- with class probabilities and the
behavioral stats (api/uniq-api/resource counts) that explain why.

Works on either split — pass "val" (default) or "test" as the first arg.
GPU BOX ONLY (mamba backbone is CUDA-only). Read-only; writes one CSV per split.
Run:  /venv/main/bin/python -u probe_missed_val.py [val|test] 2>&1 | tee ~/logs/probe_missed.log
"""
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from graph_construction.pytorch_dataset import RansomwareGraphDataset, hetero_collate
from graph_construction.vocabulary import load_vocab
from models.tgnn_model import create_hetero_tgnn
from utils.schema_manager import ensure_active_schema
# reuse the EXACT forward-prep helpers the trainer uses in validate()
from models.train import (
    _extract_timestamps,
    _extract_api_indices,
    _validate_batch_before_forward,
    _maybe_perturb_edge_attr_dict,
    compute_threat_vs_benign,
)

CLASS = {0: "Ransom", 1: "Benign", 2: "Malware"}
ROOT = Path(config.PROJECT_ROOT)

SPLIT = sys.argv[1].lower() if len(sys.argv) > 1 else "val"
assert SPLIT in ("val", "test", "train"), f"split must be val/test/train, got {SPLIT!r}"

device = torch.device(config.DEVICE)
print(f"device = {device} | backbone = {config.TEMPORAL_BACKBONE} | SPLIT = {SPLIT}")

schema = ensure_active_schema()
vocab = load_vocab(config.RESULTS_DIR / "api_vocabulary.json")
print(f"vocab APIs: {vocab.size()}")

val_dataset = RansomwareGraphDataset(
    data_dir=str(config.DATA_DIR),
    split=SPLIT,
    cache_dir=str(config.ACTIVE_GRAPH_DIR),
    schema=schema,
    validate_graphs=True,
)
graph_files = val_dataset.graph_files          # index order == loader order (shuffle=False)
print(f"{SPLIT.upper()} graphs: {len(graph_files)}")

val_loader = DataLoader(
    val_dataset,
    batch_size=config.BATCH_SIZE,
    shuffle=False,                              # <-- preserves md5<->pred mapping
    collate_fn=lambda b: hetero_collate(b, schema=schema, validate_batch=False),
    num_workers=0,                              # deterministic ordering
    pin_memory=False,
)

model = create_hetero_tgnn(use_memory=config.USE_MEMORY).to(device)
ckpt = torch.load(config.CHECKPOINTS_DIR / "best_model.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"loaded checkpoint: epoch={ckpt.get('epoch')} best_metric={ckpt.get('best_metric'):.4f}")

all_probs, all_preds, all_labels = [], [], []
with torch.no_grad():
    for batch_data, labels in val_loader:
        labels = labels.to(device)
        batch_data = batch_data.to(device)
        timestamps_dict = _extract_timestamps(batch_data)
        api_indices = _extract_api_indices(batch_data, vocab, device)
        _validate_batch_before_forward(batch_data, schema, api_indices=api_indices)
        edge_attr_dict = _maybe_perturb_edge_attr_dict(batch_data.edge_attr_dict, training=False)
        logits = model(
            x_dict=batch_data.x_dict,
            edge_index_dict=batch_data.edge_index_dict,
            edge_attr_dict=edge_attr_dict,
            batch_dict=batch_data.batch_dict,
            timestamps_dict=timestamps_dict,
            api_indices=api_indices,
            current_epoch=None,
        )
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs.cpu().numpy())
        all_preds.append(probs.argmax(1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

probs = np.concatenate(all_probs)
preds = np.concatenate(all_preds)
labels = np.concatenate(all_labels)
assert len(preds) == len(graph_files), f"order mismatch {len(preds)} vs {len(graph_files)}"

# confusion + threats-missed (same math as the trainer)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
tvb = compute_threat_vs_benign(cm)
print("\n=== CONFUSION (rows=true 0/1/2, cols=pred 0/1/2) ===")
print(cm)
print(f"\nThreats missed (threat->Benign) = {tvb['threats_missed']}  "
      f"| threat_recall={tvb['threat_recall']:.4f}  benign_recall={tvb['benign_recall']:.4f}")

# the missed: true threat (0 or 2) predicted Benign (1)
missed_idx = [i for i in range(len(preds)) if labels[i] in (0, 2) and preds[i] == 1]
print(f"\n=== {len(missed_idx)} MISSED THREATS (true malware/ransom -> predicted Benign) ===")

graph_files_by_md5 = {p.stem: p for p in graph_files}

def gstats(md5):
    g = pickle.load(open(graph_files_by_md5[md5], "rb"))
    n_api = int(g["api"].x.shape[0]) if "api" in g.node_types else 0
    names = list(getattr(g["api"], "api_names", [])) if "api" in g.node_types else []
    res = {nt: (int(g[nt].x.shape[0]) if nt in g.node_types else 0)
           for nt in ("file", "registry", "process", "network")}
    return n_api, len(set(names)), res, sorted(set(names))

rows = []
for i in missed_idx:
    md5 = graph_files[i].stem
    p = probs[i]
    n_api, uniq, res, unames = gstats(md5)
    rc = f"{res['file']}/{res['registry']}/{res['process']}/{res['network']}"
    print(f"\n  {md5}")
    print(f"    true={CLASS[labels[i]]}  pred=Benign   "
          f"P[ransom/benign/malware]={p[0]:.3f}/{p[1]:.3f}/{p[2]:.3f}  "
          f"(benign won by {p[1]-max(p[0],p[2]):+.3f})")
    print(f"    n_api={n_api}  uniq_api={uniq}  res(file/reg/proc/net)={rc}")
    print(f"    unique APIs: {unames[:20]}{' ...' if len(unames)>20 else ''}")
    rows.append([md5, CLASS[labels[i]], f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}",
                 n_api, uniq, res['file'], res['registry'], res['process'], res['network']])

# also: ALL threats the model was *unsure* about (benign prob in top-2 but didn't win) -- the margin picture
borderline = sorted(
    [i for i in range(len(preds)) if labels[i] in (0, 2) and preds[i] != 1 and probs[i, 1] > 0.20],
    key=lambda i: -probs[i, 1],
)[:10]
if borderline:
    print(f"\n=== {len(borderline)} CAUGHT-but-shaky threats (P(benign)>0.20, still classified threat) ===")
    for i in borderline:
        md5 = graph_files[i].stem
        p = probs[i]
        print(f"  {md5[:16]}  true={CLASS[labels[i]]} pred={CLASS[preds[i]]}  "
              f"P[r/b/m]={p[0]:.3f}/{p[1]:.3f}/{p[2]:.3f}")

out = ROOT / f"missed_{SPLIT}_threats.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["md5", "true", "P_ransom", "P_benign", "P_malware",
                "n_api", "uniq_api", "n_file", "n_reg", "n_proc", "n_net"])
    w.writerows(rows)
print(f"\nwrote {len(rows)} -> {out}")
print("DONE")
