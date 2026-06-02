#!/usr/bin/env python3
"""
scripts/ablate_connects_to.py

Controlled ablation: run the trained model on the `all` zero-day graphs twice.
  A = intact (network-saturated, as-is)
  B = with ('api','connects_to','network') edges removed
      (keeps network NODES but no network EDGES -> structurally matches `old`)

Everything else (checkpoint, schema normalization, collate, forward) is held
identical, so any prediction change is attributable to the network/connects_to
relation only. Each graph is run through BOTH variants in the same iteration so
any skipped (OOM/error) graph is dropped from both, keeping the comparison paired.
"""
from __future__ import annotations

import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import config
from models.tgnn_model import create_hetero_tgnn
from graph_construction.vocabulary import load_vocab
from utils.schema_manager import ensure_active_schema
from utils.schema_utils import normalize_graph_node_features

# reuse the exact eval helpers
from scripts.evaluate_zeroday import (
    custom_eval_collate,
    _extract_timestamps,
    _extract_api_indices,
)

CONNECTS_TO = ("api", "connects_to", "network")
ALL_DIR = config.CACHE_DIR / "zero_day_unseen" / "all"
TRUE_LABEL = config.CUSTOM_EVAL_NEW_MALWARE_LABEL  # 2 = Other Malwares


def get_label(g) -> int:
    try:
        y = g._global_store["y"]
        return int(y.item() if hasattr(y, "item") else y)
    except Exception:
        return TRUE_LABEL


@torch.no_grad()
def predict(model, g, schema, vocab, device, drop_connects_to: bool):
    g = pickle.loads(pickle.dumps(g))  # deep copy
    if config.NORMALIZE_NODE_FEATURE_DIMS:
        g = normalize_graph_node_features(g, schema=schema, inplace=True)
    if drop_connects_to and CONNECTS_TO in g.edge_types:
        del g[CONNECTS_TO]
    batched, labels = custom_eval_collate([(g, TRUE_LABEL)], schema=schema, validate_batch=False)
    batched = batched.to(device)
    timestamps_dict = _extract_timestamps(batched)
    api_indices = _extract_api_indices(batched, vocab, device)
    logits = model(
        x_dict=batched.x_dict,
        edge_index_dict=batched.edge_index_dict,
        edge_attr_dict=batched.edge_attr_dict if hasattr(batched, "edge_attr_dict") else None,
        batch_dict=batched.batch_dict if hasattr(batched, "batch_dict") else None,
        timestamps_dict=timestamps_dict,
        api_indices=api_indices,
        current_epoch=config.EPOCHS,
    )
    return int(logits.argmax(dim=1).item())


def report(name, labels, preds):
    labels = np.array(labels); preds = np.array(preds)
    acc = (labels == preds).mean()
    dist = Counter(preds.tolist())
    print(f"  [{name}] accuracy(=recall on class {TRUE_LABEL}) = {acc:.3f}"
          f"   pred dist: 0(Ransom)={dist.get(0,0)} 1(Benign)={dist.get(1,0)} 2(OtherMal)={dist.get(2,0)}")
    return acc


def main():
    config.set_seed()
    device = torch.device(config.DEVICE)
    schema = ensure_active_schema()
    vocab = load_vocab(config.RESULTS_DIR / "api_vocabulary.json")

    model = create_hetero_tgnn(use_memory=config.USE_MEMORY).to(device)
    ckpt = torch.load(config.CUSTOM_EVAL_CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    print(f"[INFO] device={device}  checkpoint epoch={ckpt.get('epoch')}")

    files = sorted(ALL_DIR.glob("*.pkl"))
    print(f"[INFO] {len(files)} graphs in {ALL_DIR}\n")

    labels, preds_intact, preds_dropped, flips, skipped = [], [], [], [], 0
    for i, f in enumerate(files):
        with open(f, "rb") as fh:
            g = pickle.load(fh)
        lab = get_label(g)
        try:
            p_a = predict(model, g, schema, vocab, device, drop_connects_to=False)
            p_b = predict(model, g, schema, vocab, device, drop_connects_to=True)
        except Exception as e:
            skipped += 1
            print(f"  ! skip {f.name[:40]} ({type(e).__name__}: {str(e)[:60]})")
            continue
        labels.append(lab); preds_intact.append(p_a); preds_dropped.append(p_b)
        if p_a != p_b:
            flips.append((f.name, p_a, p_b))
        print(f"  [{i+1}/{len(files)}] {f.name[:34]} y={lab} intact->{p_a} dropCT->{p_b}"
              f"{'   <-- FLIP' if p_a != p_b else ''}")

    print(f"\n=== RESULTS (n={len(labels)}, skipped={skipped}) ===")
    a = report("A intact (network-saturated)", labels, preds_intact)
    b = report("B drop connects_to (matches `old` structure)", labels, preds_dropped)
    print(f"\n  prediction flips when removing network edges: {len(flips)}")
    for name, pa, pb in flips:
        print(f"    {name[:40]}: {pa} -> {pb}")
    print(f"\n  delta accuracy (B - A) = {b - a:+.3f}")
    if b > a:
        print("  => Removing the network/connects_to relation IMPROVES predictions.")
        print("     Confirms the network regime is corrupting the readout (root cause).")
    elif b < a:
        print("  => Removing network edges HURTS. Network signal is helping; cause is elsewhere.")
    else:
        print("  => No change. The connects_to relation is not the driver; cause is elsewhere")
        print("     (likely the temporal/sequence content of these specific zero-day samples).")


if __name__ == "__main__":
    main()
