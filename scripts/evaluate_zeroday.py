# #!/usr/bin/env python3
# """
# scripts/evaluate_zeroday.py

# Evaluate an already-trained TGNN checkpoint on a custom zero-day / mixed
# evaluation set built on the fly from existing graph folders.

# Selection policy from config.py: 
#     - 200 benign from ACTIVE_GRAPH_DIR/{val,test}
#     - all new unseen malware from CUSTOM_EVAL_NEW_MALWARE_DIR
#     - 55 old malware from ACTIVE_GRAPH_DIR/{val,test}
#     - all ransomware from ACTIVE_GRAPH_DIR/{val,test}

# This script mirrors train.py validation/test behavior as closely as possible:
#     - same schema setup
#     - same graph preprocessing path as RansomwareGraphDataset.__getitem__()
#     - same hetero_collate behavior
#     - same model forward path
#     - same focal loss
#     - same checkpoint loading style
#     - same val/test-style DataLoader batching using config.BATCH_SIZE

# Outputs:
#     - manifest CSV
#     - predictions CSV
#     - results JSON
# """

# from __future__ import annotations

# import json
# import pickle
# import random
# import sys
# from collections import Counter
# from functools import partial
# from pathlib import Path
# from typing import Dict, List

# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
# from torch.utils.data import Dataset, DataLoader

# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# sys.path.append(str(PROJECT_ROOT))

# import config
# from graph_construction.pytorch_dataset import hetero_collate, RansomwareGraphDataset
# from graph_construction.vocabulary import load_vocab
# from models.memory_utils import get_api_indices
# from models.tgnn_model import create_hetero_tgnn
# from utils.schema_manager import ensure_active_schema
# from utils.schema_utils import normalize_graph_node_features
# from utils.schema_validator import validate_batch_or_raise, validate_graph_or_raise


# # ============================================================
# # Loss (same as train.py)
# # ============================================================

# class FocalLoss(nn.Module):
#     def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
#         super().__init__()
#         self.gamma = gamma
#         self.reduction = reduction
#         if alpha is not None:
#             self.alpha = torch.tensor(alpha) if isinstance(alpha, list) else alpha
#         else:
#             self.alpha = None

#     def forward(self, inputs, targets):
#         probs = F.softmax(inputs, dim=1)
#         class_mask = F.one_hot(targets, num_classes=inputs.size(1)).float()
#         probs = (probs * class_mask).sum(dim=1)
#         focal_w = (1 - probs) ** self.gamma
#         ce_loss = F.cross_entropy(inputs, targets, reduction="none")
#         loss = focal_w * ce_loss

#         if self.alpha is not None:
#             if self.alpha.device != inputs.device:
#                 self.alpha = self.alpha.to(inputs.device)
#             loss = self.alpha[targets] * loss

#         return loss.mean() if self.reduction == "mean" else loss.sum()


# # ============================================================
# # Label helpers
# # ============================================================

# def category_to_label(all_categories: str) -> int:
#     cats = str(all_categories).strip().lower()
#     if "ransomware" in cats:
#         return 0
#     elif "benign" in cats:
#         return 1
#     else:
#         return 2


# def label_to_name(label: int) -> str:
#     return config.CLASS_NAMES.get(int(label), f"Class_{label}")


# # ============================================================
# # Metrics (same style as train.py)
# # ============================================================

# def compute_metrics(predictions, labels):
#     acc = accuracy_score(labels, predictions)
#     precision, recall, f1, support = precision_recall_fscore_support(
#         labels, predictions, average=None, zero_division=0, labels=[0, 1, 2]
#     )
#     macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
#         labels, predictions, average="macro", zero_division=0, labels=[0, 1, 2]
#     )
#     w_p, w_r, w_f1, _ = precision_recall_fscore_support(
#         labels, predictions, average="weighted", zero_division=0, labels=[0, 1, 2]
#     )
#     cm = confusion_matrix(labels, predictions, labels=[0, 1, 2])

#     return {
#         "accuracy": float(acc),
#         "per_class_precision": precision.tolist(),
#         "per_class_recall": recall.tolist(),
#         "per_class_f1": f1.tolist(),
#         "per_class_support": support.tolist(),
#         "macro_precision": float(macro_p),
#         "macro_recall": float(macro_r),
#         "macro_f1": float(macro_f1),
#         "weighted_precision": float(w_p),
#         "weighted_recall": float(w_r),
#         "weighted_f1": float(w_f1),
#         "confusion_matrix": cm.tolist(),
#     }


# def compute_threat_vs_benign(confusion_mat: np.ndarray) -> dict:
#     TP = confusion_mat[0, 0] + confusion_mat[0, 2] + confusion_mat[2, 0] + confusion_mat[2, 2]
#     FN = confusion_mat[0, 1] + confusion_mat[2, 1]
#     TN = confusion_mat[1, 1]
#     FP = confusion_mat[1, 0] + confusion_mat[1, 2]

#     threat_recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
#     threat_precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
#     threat_f1 = (
#         2 * threat_precision * threat_recall / (threat_precision + threat_recall)
#         if (threat_precision + threat_recall) > 0 else 0.0
#     )

#     benign_recall = TN / (TN + FP) if (TN + FP) > 0 else 0.0
#     benign_precision = TN / (TN + FN) if (TN + FN) > 0 else 0.0
#     benign_f1 = (
#         2 * benign_precision * benign_recall / (benign_precision + benign_recall)
#         if (benign_precision + benign_recall) > 0 else 0.0
#     )

#     accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

#     return {
#         "TP": int(TP),
#         "FN": int(FN),
#         "TN": int(TN),
#         "FP": int(FP),
#         "threat_recall": float(threat_recall),
#         "threat_precision": float(threat_precision),
#         "threat_f1": float(threat_f1),
#         "benign_recall": float(benign_recall),
#         "benign_precision": float(benign_precision),
#         "benign_f1": float(benign_f1),
#         "accuracy_2class": float(accuracy),
#         "threats_missed": int(FN),
#     }


# def print_metrics(metrics, split="Custom Eval"):
#     print(f"\n{'=' * 72}")
#     print(f"{split} Metrics")
#     print(f"{'=' * 72}")
#     print(f"Accuracy:    {metrics['accuracy']:.4f}")
#     print(f"Macro F1:    {metrics['macro_f1']:.4f}")
#     print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
#     print(f"Loss:        {metrics.get('loss', 0.0):.4f}")
#     print("\nPer-Class Metrics")
#     print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<12}")
#     print(f"{'-' * 72}")
#     for i in range(3):
#         print(
#             f"{config.CLASS_NAMES[i]:<20} "
#             f"{metrics['per_class_precision'][i]:<12.4f} "
#             f"{metrics['per_class_recall'][i]:<12.4f} "
#             f"{metrics['per_class_f1'][i]:<12.4f} "
#             f"{int(metrics['per_class_support'][i]):<12}"
#         )
#     print("\nConfusion Matrix")
#     print(np.array(metrics["confusion_matrix"]))
#     print(f"{'=' * 72}")


# # ============================================================
# # Manifest building
# # ============================================================

# def load_metadata_labels() -> Dict[str, int]:
#     df = pd.read_csv(config.METADATA_PATH)

#     if "md5" not in df.columns:
#         raise KeyError("Expected 'md5' column in metadata CSV.")
#     if "all_categories" not in df.columns:
#         raise KeyError("Expected 'all_categories' column in metadata CSV.")

#     df["_md5"] = df["md5"].astype(str).str.strip().str.lower()

#     return {
#         row["_md5"]: category_to_label(row["all_categories"])
#         for _, row in df.iterrows()
#     }


# def discover_old_split_graphs(label_map: Dict[str, int]) -> List[dict]:
#     rows = []

#     for split in config.CUSTOM_EVAL_SOURCE_SPLITS:
#         split_dir = Path(config.ACTIVE_GRAPH_DIR) / split
#         if not split_dir.exists():
#             raise FileNotFoundError(f"Missing split dir: {split_dir}")

#         for graph_path in sorted(split_dir.glob("*.pkl")):
#             md5 = graph_path.stem.strip().lower()
#             if md5 not in label_map:
#                 continue

#             label = int(label_map[md5])
#             rows.append({
#                 "sample_id": md5,
#                 "graph_path": str(graph_path.resolve()),
#                 "label": label,
#                 "class_name": label_to_name(label),
#                 "origin": f"old_{split}",
#                 "is_new_unseen": False,
#             })

#     return rows


# def discover_new_unseen_graphs(old_sample_ids: set[str]) -> List[dict]:
#     new_dir = Path(config.CUSTOM_EVAL_NEW_MALWARE_DIR)

#     if not new_dir.exists():
#         if config.CUSTOM_EVAL_FAIL_IF_NEW_MALWARE_DIR_MISSING:
#             raise FileNotFoundError(f"Missing CUSTOM_EVAL_NEW_MALWARE_DIR: {new_dir}")
#         print(f"[WARN] New malware dir missing: {new_dir}")
#         return []

#     rows = []
#     overlap = []

#     for graph_path in sorted(new_dir.glob("*.pkl")):
#         sample_id = graph_path.stem.strip().lower()

#         if config.CUSTOM_EVAL_ENFORCE_NEW_UNSEEN_DISJOINT and sample_id in old_sample_ids:
#             overlap.append(sample_id)
#             continue

#         rows.append({
#             "sample_id": sample_id,
#             "graph_path": str(graph_path.resolve()),
#             "label": int(config.CUSTOM_EVAL_NEW_MALWARE_LABEL),
#             "class_name": label_to_name(int(config.CUSTOM_EVAL_NEW_MALWARE_LABEL)),
#             "origin": "new_unseen",
#             "is_new_unseen": True,
#         })

#     if overlap:
#         raise RuntimeError(
#             "Overlap detected between new unseen malware and old val/test pools. "
#             f"Count={len(overlap)}. Example={overlap[:10]}"
#         )

#     return rows


# def sample_rows(rows: List[dict], n: int, name: str) -> List[dict]:
#     if len(rows) < n:
#         msg = f"Requested {n} {name} samples, but only found {len(rows)}."
#         if name == "benign" and config.CUSTOM_EVAL_FAIL_IF_NOT_ENOUGH_BENIGN:
#             raise RuntimeError(msg)
#         if name == "old_malware" and config.CUSTOM_EVAL_FAIL_IF_NOT_ENOUGH_OLD_MALWARE:
#             raise RuntimeError(msg)
#         print(f"[WARN] {msg} Using all available.")
#         return list(rows)

#     rows = list(rows)
#     if config.CUSTOM_EVAL_SHUFFLE_BEFORE_SAMPLE:
#         rng = random.Random(config.RANDOM_SEED)
#         rng.shuffle(rows)

#     return rows[:n]


# def build_manifest() -> pd.DataFrame:
#     random.seed(config.RANDOM_SEED)
#     np.random.seed(config.RANDOM_SEED)

#     label_map = load_metadata_labels()
#     old_rows = discover_old_split_graphs(label_map)
#     old_sample_ids = {r["sample_id"] for r in old_rows}
#     new_rows = discover_new_unseen_graphs(old_sample_ids)

#     benign_pool = [r for r in old_rows if r["label"] == 1]
#     ransomware_pool = [r for r in old_rows if r["label"] == 0]
#     old_malware_pool = [r for r in old_rows if r["label"] == 2]

#     selected_benign = sample_rows(benign_pool, int(config.CUSTOM_EVAL_TARGET_BENIGN), "benign")
#     selected_old_malware = sample_rows(old_malware_pool, int(config.CUSTOM_EVAL_TARGET_OLD_MALWARE), "old_malware")
#     selected_ransomware = ransomware_pool if config.CUSTOM_EVAL_INCLUDE_ALL_RANSOMWARE else []
#     selected_new_malware = new_rows if config.CUSTOM_EVAL_INCLUDE_ALL_NEW_MALWARE else []

#     final_rows = selected_benign + selected_old_malware + selected_new_malware + selected_ransomware
#     if not final_rows:
#         raise RuntimeError("Custom evaluation manifest is empty.")

#     df = pd.DataFrame(final_rows).drop_duplicates(subset=["graph_path"]).reset_index(drop=True)
#     df["label"] = df["label"].astype(int)
#     df["class_name"] = df["label"].map(label_to_name)

#     config.CUSTOM_EVAL_DIR.mkdir(parents=True, exist_ok=True)
#     df.to_csv(config.CUSTOM_EVAL_MANIFEST_PATH, index=False)

#     print("\n[INFO] Custom manifest created:")
#     print(f"       path   : {config.CUSTOM_EVAL_MANIFEST_PATH}")
#     print(f"       total  : {len(df)}")
#     print(f"       benign : {(df['label'] == 1).sum()}")
#     print(f"       ransom : {(df['label'] == 0).sum()}")
#     print(f"       other  : {(df['label'] == 2).sum()}")
#     print(f"       origin : {dict(Counter(df['origin'].tolist()))}")

#     return df


# # ============================================================
# # Dataset
# # ============================================================

# class CustomManifestDataset(Dataset):
#     """
#     Custom manifest dataset that mirrors RansomwareGraphDataset.__getitem__()
#     as closely as possible, but loads graph paths from a manifest instead of
#     split folders.
#     """

#     def __init__(
#         self,
#         manifest_csv: str | Path,
#         schema: Dict,
#         graph_processor,
#         validate_graphs: bool = True,
#     ):
#         self.manifest_path = Path(manifest_csv)
#         self.df = pd.read_csv(self.manifest_path)
#         self.schema = schema
#         self.graph_processor = graph_processor
#         self.validate_graphs = validate_graphs

#         required = {"graph_path", "label", "sample_id"}
#         missing = required - set(self.df.columns)
#         if missing:
#             raise KeyError(f"Manifest missing required columns: {missing}")

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx: int):
#         row = self.df.iloc[idx]
#         graph_path = Path(row["graph_path"])
#         label = int(row["label"])
#         sample_id = str(row["sample_id"])

#         if not graph_path.exists():
#             raise FileNotFoundError(f"Graph not found: {graph_path}")

#         with open(graph_path, "rb") as f:
#             graph = pickle.load(f)

#         graph = self.graph_processor._ensure_api_timestamps(graph)

#         if config.NORMALIZE_NODE_FEATURE_DIMS:
#             graph = normalize_graph_node_features(graph, schema=self.schema, inplace=True)

#         graph = self.graph_processor._compress_consecutive_api_runs(graph)
#         graph = self.graph_processor._behavior_aware_api_cap(graph)
#         graph = self.graph_processor._ensure_api_aux_tensors(graph)

#         if self.validate_graphs and config.FAIL_ON_SCHEMA_MISMATCH:
#             validate_graph_or_raise(
#                 graph,
#                 schema=self.schema,
#                 require_api_nodes=True,
#                 strict=config.SCHEMA_STRICT_VALIDATION,
#             )

#         graph.sample_id = sample_id
#         graph.graph_path = str(graph_path)

#         return graph, label


# # ============================================================
# # Helpers (same as train.py)
# # ============================================================

# def _extract_timestamps(batch_data) -> dict:
#     try:
#         graph_timestamps = object.__getattribute__(batch_data, "graph_timestamps")
#         if graph_timestamps is not None:
#             return graph_timestamps
#     except (AttributeError, KeyError):
#         pass

#     ts = {}
#     try:
#         for nt in batch_data.x_dict.keys():
#             if hasattr(batch_data[nt], "timestamps"):
#                 ts[nt] = batch_data[nt].timestamps
#     except (KeyError, AttributeError):
#         pass

#     return ts or None


# def _extract_api_indices(batch_data, vocab, device):
#     try:
#         api_names = batch_data["api"].api_names
#         return get_api_indices(api_names, vocab, device=device)
#     except (AttributeError, KeyError):
#         n = batch_data["api"].x.size(0)
#         return torch.zeros(n, dtype=torch.long, device=device)


# def _validate_batch_before_forward(batch_data, schema, api_indices=None):
#     edge_attr_dict = batch_data.edge_attr_dict if hasattr(batch_data, "edge_attr_dict") else None
#     batch_dict = batch_data.batch_dict if hasattr(batch_data, "batch_dict") else None

#     validate_batch_or_raise(
#         x_dict=batch_data.x_dict,
#         edge_index_dict=batch_data.edge_index_dict,
#         edge_attr_dict=edge_attr_dict,
#         batch_dict=batch_dict,
#         schema=schema,
#         require_api=True,
#         require_api_indices_if_memory=(api_indices is not None),
#         api_indices=api_indices,
#     )


# def custom_eval_collate(batch, schema, validate_batch=False):
#     """
#     Same hetero_collate path as train.py, but preserve sample-level metadata
#     for prediction export.
#     """
#     if not batch:
#         return None, None

#     sample_ids = []
#     graph_paths = []

#     for graph, _label in batch:
#         sample_ids.append(getattr(graph, "sample_id", ""))
#         graph_paths.append(getattr(graph, "graph_path", ""))

#     batched_graph, labels = hetero_collate(
#         batch,
#         schema=schema,
#         validate_batch=validate_batch,
#     )

#     if batched_graph is None or labels is None:
#         return batched_graph, labels

#     batched_graph.sample_id = sample_ids
#     batched_graph.graph_path = graph_paths
#     return batched_graph, labels


# # ============================================================
# # Eval loop (same style as train.py validate)
# # ============================================================

# @torch.no_grad()
# def run_eval(model, loader, criterion, vocab, schema, device, eval_epoch=None):
#     model.eval()

#     total_loss = 0.0
#     all_preds = []
#     all_labels = []
#     prediction_rows = []

#     for batch_item in loader:
#         if batch_item is None:
#             continue

#         if not isinstance(batch_item, (tuple, list)) or len(batch_item) != 2:
#             continue

#         batch_data, labels = batch_item
#         if batch_data is None or labels is None:
#             continue

#         labels = labels.to(device)
#         batch_data = batch_data.to(device)

#         timestamps_dict = _extract_timestamps(batch_data)
#         api_indices = _extract_api_indices(batch_data, vocab, device)

#         _validate_batch_before_forward(batch_data, schema, api_indices=api_indices)

#         logits = model(
#             x_dict=batch_data.x_dict,
#             edge_index_dict=batch_data.edge_index_dict,
#             edge_attr_dict=batch_data.edge_attr_dict if hasattr(batch_data, "edge_attr_dict") else None,
#             batch_dict=batch_data.batch_dict if hasattr(batch_data, "batch_dict") else None,
#             timestamps_dict=timestamps_dict,
#             api_indices=api_indices,
#             current_epoch=eval_epoch,
#         )

#         loss = criterion(logits, labels)
#         total_loss += float(loss.item())

#         probs = torch.softmax(logits, dim=1)
#         preds = logits.argmax(dim=1)

#         all_preds.extend(preds.cpu().numpy().tolist())
#         all_labels.extend(labels.cpu().numpy().tolist())

#         sample_ids = getattr(batch_data, "sample_id", None)
#         graph_paths = getattr(batch_data, "graph_path", None)

#         if sample_ids is None:
#             sample_ids = [f"sample_{i}" for i in range(labels.size(0))]
#         if graph_paths is None:
#             graph_paths = [""] * labels.size(0)

#         for i in range(labels.size(0)):
#             prediction_rows.append({
#                 "sample_id": sample_ids[i] if isinstance(sample_ids, list) else str(sample_ids),
#                 "graph_path": graph_paths[i] if isinstance(graph_paths, list) else str(graph_paths),
#                 "true_label": int(labels[i].item()),
#                 "true_name": label_to_name(int(labels[i].item())),
#                 "pred_label": int(preds[i].item()),
#                 "pred_name": label_to_name(int(preds[i].item())),
#                 "prob_ransomware": float(probs[i, 0].item()),
#                 "prob_benign": float(probs[i, 1].item()),
#                 "prob_other_malware": float(probs[i, 2].item()),
#                 "correct": int(preds[i].item() == labels[i].item()),
#             })

#     if len(all_labels) == 0:
#         raise RuntimeError("No evaluation samples were processed.")

#     metrics = compute_metrics(all_preds, all_labels)
#     metrics["loss"] = total_loss / max(len(loader), 1)
#     return metrics, prediction_rows


# # ============================================================
# # Main
# # ============================================================

# def main():
#     if not config.USE_CUSTOM_ZERO_DAY_EVAL:
#         print("USE_CUSTOM_ZERO_DAY_EVAL is False in config.py. Exiting.")
#         return

#     config.set_seed()
#     device = torch.device(config.DEVICE)
#     print(f"[INFO] Device: {device}")

#     schema = ensure_active_schema()
#     if isinstance(schema, dict):
#         active_schema = schema
#     else:
#         with open(config.ACTIVE_SCHEMA_PATH, "r", encoding="utf-8") as f:
#             active_schema = json.load(f)

#     if config.CUSTOM_EVAL_REBUILD_MANIFEST or not Path(config.CUSTOM_EVAL_MANIFEST_PATH).exists():
#         manifest_df = build_manifest()
#     else:
#         manifest_df = pd.read_csv(config.CUSTOM_EVAL_MANIFEST_PATH)
#         print(f"[INFO] Using existing manifest: {config.CUSTOM_EVAL_MANIFEST_PATH}")

#     graph_processor = RansomwareGraphDataset.__new__(RansomwareGraphDataset)
#     graph_processor.schema = active_schema

#     dataset = CustomManifestDataset(
#         manifest_csv=config.CUSTOM_EVAL_MANIFEST_PATH,
#         schema=active_schema,
#         graph_processor=graph_processor,
#         validate_graphs=getattr(config, "CUSTOM_EVAL_VALIDATE_GRAPHS", True),
#     )

#     collate = partial(
#         custom_eval_collate,
#         schema=active_schema,
#         validate_batch=False,
#     )

#     # Mirror train.py val/test loader behavior
#     loader = DataLoader(
#         dataset,
#         batch_size=config.BATCH_SIZE,
#         shuffle=False,
#         collate_fn=collate,
#         num_workers=config.NUM_WORKERS,
#         pin_memory=config.PIN_MEMORY,
#     )

#     vocab = load_vocab(config.RESULTS_DIR / "api_vocabulary.json")

#     model = create_hetero_tgnn(use_memory=config.USE_MEMORY).to(device)

#     checkpoint_path = Path(config.CUSTOM_EVAL_CHECKPOINT_PATH)
#     if not checkpoint_path.exists():
#         raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

#     ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

#     missing = model.load_state_dict(ckpt["model_state_dict"], strict=False)
#     print("[INFO] Loaded checkpoint with strict=False")
#     if hasattr(missing, "missing_keys") and missing.missing_keys:
#         print(f"[INFO] Missing keys    : {missing.missing_keys}")
#     if hasattr(missing, "unexpected_keys") and missing.unexpected_keys:
#         print(f"[INFO] Unexpected keys : {missing.unexpected_keys}")

#     eval_epoch = int(ckpt.get("epoch", config.EPOCHS))

#     print(f"[INFO] Loaded checkpoint : {checkpoint_path}")
#     print(f"[INFO] Checkpoint epoch  : {eval_epoch}")
#     print(f"[INFO] Eval batch size   : {config.BATCH_SIZE}")

#     criterion = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)

#     metrics, prediction_rows = run_eval(
#         model=model,
#         loader=loader,
#         criterion=criterion,
#         vocab=vocab,
#         schema=active_schema,
#         device=device,
#         eval_epoch=eval_epoch,
#     )

#     print_metrics(metrics, split="Custom Zero-Day Eval")

#     cm = np.array(metrics["confusion_matrix"])
#     tvb = compute_threat_vs_benign(cm)

#     print("\nThreat vs Benign")
#     print(f"  Threat Precision : {tvb['threat_precision']:.4f}")
#     print(f"  Threat Recall    : {tvb['threat_recall']:.4f}")
#     print(f"  Threat F1        : {tvb['threat_f1']:.4f}")
#     print(f"  Benign Precision : {tvb['benign_precision']:.4f}")
#     print(f"  Benign Recall    : {tvb['benign_recall']:.4f}")
#     print(f"  Benign F1        : {tvb['benign_f1']:.4f}")
#     print(f"  Threats Missed   : {tvb['threats_missed']}")

#     pred_df = pd.DataFrame(prediction_rows)
#     pred_df.to_csv(config.CUSTOM_EVAL_PREDICTIONS_PATH, index=False)

#     results = {
#         "manifest_path": str(config.CUSTOM_EVAL_MANIFEST_PATH),
#         "checkpoint_path": str(checkpoint_path),
#         "checkpoint_epoch": eval_epoch,
#         "num_samples": int(len(manifest_df)),
#         "eval_batch_size": int(config.BATCH_SIZE),
#         "label_distribution": {
#             "ransomware": int((manifest_df["label"] == 0).sum()),
#             "benign": int((manifest_df["label"] == 1).sum()),
#             "other_malware": int((manifest_df["label"] == 2).sum()),
#         },
#         "origin_distribution": {
#             k: int(v) for k, v in Counter(manifest_df["origin"].tolist()).items()
#         },
#         "metrics_3class": metrics,
#         "metrics_threat_vs_benign": tvb,
#     }

#     with open(config.CUSTOM_EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
#         json.dump(results, f, indent=2)

#     print(f"\n[INFO] Saved manifest    : {config.CUSTOM_EVAL_MANIFEST_PATH}")
#     print(f"[INFO] Saved predictions : {config.CUSTOM_EVAL_PREDICTIONS_PATH}")
#     print(f"[INFO] Saved results     : {config.CUSTOM_EVAL_RESULTS_PATH}")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
scripts/evaluate_zeroday.py

Evaluate an already-trained TGNN checkpoint on a custom zero-day / mixed
evaluation set built on the fly from existing graph folders.

Adds:
    - semantic / structural / temporal branch-weight export
    - dominant branch per sample
    - branch usage summary in terminal + JSON

Requires:
    config.USE_ADAPTIVE_BRANCH_FUSION = True
"""

from __future__ import annotations

import json
import pickle
import random
import sys
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import config
from graph_construction.pytorch_dataset import hetero_collate, RansomwareGraphDataset
from graph_construction.vocabulary import load_vocab
from models.memory_utils import get_api_indices
from models.tgnn_model import create_hetero_tgnn
from utils.schema_manager import ensure_active_schema
from utils.schema_utils import normalize_graph_node_features
from utils.schema_validator import validate_batch_or_raise, validate_graph_or_raise


BRANCH_NAMES = ["semantic", "structural", "temporal"]


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        if alpha is not None:
            self.alpha = torch.tensor(alpha) if isinstance(alpha, list) else alpha
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)
        class_mask = F.one_hot(targets, num_classes=inputs.size(1)).float()
        probs = (probs * class_mask).sum(dim=1)
        focal_w = (1 - probs) ** self.gamma
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        loss = focal_w * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            loss = self.alpha[targets] * loss

        return loss.mean() if self.reduction == "mean" else loss.sum()


def category_to_label(all_categories: str) -> int:
    cats = str(all_categories).strip().lower()
    if "ransomware" in cats:
        return 0
    elif "benign" in cats:
        return 1
    else:
        return 2


def label_to_name(label: int) -> str:
    return config.CLASS_NAMES.get(int(label), f"Class_{label}")


def compute_metrics(predictions, labels):
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0, labels=[0, 1, 2]
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0, labels=[0, 1, 2]
    )
    w_p, w_r, w_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0, labels=[0, 1, 2]
    )
    cm = confusion_matrix(labels, predictions, labels=[0, 1, 2])

    return {
        "accuracy": float(acc),
        "per_class_precision": precision.tolist(),
        "per_class_recall": recall.tolist(),
        "per_class_f1": f1.tolist(),
        "per_class_support": support.tolist(),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(w_p),
        "weighted_recall": float(w_r),
        "weighted_f1": float(w_f1),
        "confusion_matrix": cm.tolist(),
    }


def compute_threat_vs_benign(confusion_mat: np.ndarray) -> dict:
    TP = confusion_mat[0, 0] + confusion_mat[0, 2] + confusion_mat[2, 0] + confusion_mat[2, 2]
    FN = confusion_mat[0, 1] + confusion_mat[2, 1]
    TN = confusion_mat[1, 1]
    FP = confusion_mat[1, 0] + confusion_mat[1, 2]

    threat_recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    threat_precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    threat_f1 = (
        2 * threat_precision * threat_recall / (threat_precision + threat_recall)
        if (threat_precision + threat_recall) > 0 else 0.0
    )

    benign_recall = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    benign_precision = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    benign_f1 = (
        2 * benign_precision * benign_recall / (benign_precision + benign_recall)
        if (benign_precision + benign_recall) > 0 else 0.0
    )

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    return {
        "TP": int(TP),
        "FN": int(FN),
        "TN": int(TN),
        "FP": int(FP),
        "threat_recall": float(threat_recall),
        "threat_precision": float(threat_precision),
        "threat_f1": float(threat_f1),
        "benign_recall": float(benign_recall),
        "benign_precision": float(benign_precision),
        "benign_f1": float(benign_f1),
        "accuracy_2class": float(accuracy),
        "threats_missed": int(FN),
    }


def print_metrics(metrics, split="Custom Eval"):
    print(f"\n{'=' * 72}")
    print(f"{split} Metrics")
    print(f"{'=' * 72}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Loss:        {metrics.get('loss', 0.0):.4f}")
    print("\nPer-Class Metrics")
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<12}")
    print(f"{'-' * 72}")
    for i in range(3):
        print(
            f"{config.CLASS_NAMES[i]:<20} "
            f"{metrics['per_class_precision'][i]:<12.4f} "
            f"{metrics['per_class_recall'][i]:<12.4f} "
            f"{metrics['per_class_f1'][i]:<12.4f} "
            f"{int(metrics['per_class_support'][i]):<12}"
        )
    print("\nConfusion Matrix")
    print(np.array(metrics["confusion_matrix"]))
    print(f"{'=' * 72}")


def _get_branch_weights_from_model(model, batch_size: int):
    weights = getattr(model, "_last_branch_weights", None)

    if weights is None:
        return None
    if not isinstance(weights, torch.Tensor):
        return None
    if weights.dim() != 2 or weights.size(1) != 3:
        return None
    if weights.size(0) != batch_size:
        return None

    return weights.detach().float().cpu()


def summarize_branch_usage(prediction_rows: List[dict]) -> dict:
    rows = [
        r for r in prediction_rows
        if r.get("branch_semantic_weight") is not None
    ]

    if not rows:
        return {
            "available": False,
            "reason": "No branch weights found. Check USE_ADAPTIVE_BRANCH_FUSION=True.",
        }

    df = pd.DataFrame(rows)

    summary = {
        "available": True,
        "overall_mean_weights": {
            "semantic": float(df["branch_semantic_weight"].mean()),
            "structural": float(df["branch_structural_weight"].mean()),
            "temporal": float(df["branch_temporal_weight"].mean()),
        },
        "dominant_branch_counts": {
            k: int(v) for k, v in df["dominant_branch"].value_counts().to_dict().items()
        },
        "dominant_branch_percent": {
            k: float(v)
            for k, v in (df["dominant_branch"].value_counts(normalize=True) * 100).to_dict().items()
        },
        "by_predicted_class": {},
        "by_true_class": {},
        "correct_vs_wrong": {},
        "threat_vs_benign_groups": {},
    }

    for pred_label, group in df.groupby("pred_label"):
        summary["by_predicted_class"][label_to_name(int(pred_label))] = {
            "count": int(len(group)),
            "semantic_mean": float(group["branch_semantic_weight"].mean()),
            "structural_mean": float(group["branch_structural_weight"].mean()),
            "temporal_mean": float(group["branch_temporal_weight"].mean()),
            "dominant_branch_counts": {
                k: int(v) for k, v in group["dominant_branch"].value_counts().to_dict().items()
            },
        }

    for true_label, group in df.groupby("true_label"):
        summary["by_true_class"][label_to_name(int(true_label))] = {
            "count": int(len(group)),
            "semantic_mean": float(group["branch_semantic_weight"].mean()),
            "structural_mean": float(group["branch_structural_weight"].mean()),
            "temporal_mean": float(group["branch_temporal_weight"].mean()),
            "dominant_branch_counts": {
                k: int(v) for k, v in group["dominant_branch"].value_counts().to_dict().items()
            },
        }

    for correct_value, group in df.groupby("correct"):
        key = "correct" if int(correct_value) == 1 else "wrong"
        summary["correct_vs_wrong"][key] = {
            "count": int(len(group)),
            "semantic_mean": float(group["branch_semantic_weight"].mean()),
            "structural_mean": float(group["branch_structural_weight"].mean()),
            "temporal_mean": float(group["branch_temporal_weight"].mean()),
            "dominant_branch_counts": {
                k: int(v) for k, v in group["dominant_branch"].value_counts().to_dict().items()
            },
        }

    df["true_is_threat"] = df["true_label"].isin([0, 2])
    df["pred_is_threat"] = df["pred_label"].isin([0, 2])

    groups = {
        "true_threat_pred_threat": df[(df["true_is_threat"]) & (df["pred_is_threat"])],
        "true_threat_pred_benign_missed": df[(df["true_is_threat"]) & (~df["pred_is_threat"])],
        "true_benign_pred_benign": df[(~df["true_is_threat"]) & (~df["pred_is_threat"])],
        "true_benign_pred_threat_false_alarm": df[(~df["true_is_threat"]) & (df["pred_is_threat"])],
    }

    for name, group in groups.items():
        if len(group) == 0:
            summary["threat_vs_benign_groups"][name] = {
                "count": 0,
                "semantic_mean": None,
                "structural_mean": None,
                "temporal_mean": None,
                "dominant_branch_counts": {},
            }
        else:
            summary["threat_vs_benign_groups"][name] = {
                "count": int(len(group)),
                "semantic_mean": float(group["branch_semantic_weight"].mean()),
                "structural_mean": float(group["branch_structural_weight"].mean()),
                "temporal_mean": float(group["branch_temporal_weight"].mean()),
                "dominant_branch_counts": {
                    k: int(v) for k, v in group["dominant_branch"].value_counts().to_dict().items()
                },
            }

    return summary


def print_branch_usage_summary(branch_summary: dict):
    print("\nBranch Usage: Semantic vs Structural vs Temporal")
    print("=" * 72)

    if not branch_summary.get("available", False):
        print(branch_summary.get("reason", "Branch weights unavailable."))
        print("=" * 72)
        return

    overall = branch_summary["overall_mean_weights"]
    print("Overall mean branch weights:")
    print(f"  Semantic   : {overall['semantic']:.4f}")
    print(f"  Structural : {overall['structural']:.4f}")
    print(f"  Temporal   : {overall['temporal']:.4f}")

    print("\nDominant branch counts:")
    for branch, count in branch_summary["dominant_branch_counts"].items():
        pct = branch_summary["dominant_branch_percent"].get(branch, 0.0)
        print(f"  {branch:<10}: {count:<6} ({pct:.2f}%)")

    print("\nBy predicted class:")
    for class_name, stats in branch_summary["by_predicted_class"].items():
        print(
            f"  {class_name:<20} "
            f"sem={stats['semantic_mean']:.4f}  "
            f"struct={stats['structural_mean']:.4f}  "
            f"temp={stats['temporal_mean']:.4f}"
        )

    print("\nThreat-vs-benign groups:")
    for group_name, stats in branch_summary["threat_vs_benign_groups"].items():
        if stats["count"] == 0:
            print(f"  {group_name:<35} count=0")
        else:
            print(
                f"  {group_name:<35} "
                f"count={stats['count']:<5} "
                f"sem={stats['semantic_mean']:.4f}  "
                f"struct={stats['structural_mean']:.4f}  "
                f"temp={stats['temporal_mean']:.4f}"
            )

    print("=" * 72)


def load_metadata_labels() -> Dict[str, int]:
    df = pd.read_csv(config.METADATA_PATH)

    if "md5" not in df.columns:
        raise KeyError("Expected 'md5' column in metadata CSV.")
    if "all_categories" not in df.columns:
        raise KeyError("Expected 'all_categories' column in metadata CSV.")

    df["_md5"] = df["md5"].astype(str).str.strip().str.lower()

    return {
        row["_md5"]: category_to_label(row["all_categories"])
        for _, row in df.iterrows()
    }


def discover_old_split_graphs(label_map: Dict[str, int]) -> List[dict]:
    rows = []

    for split in config.CUSTOM_EVAL_SOURCE_SPLITS:
        split_dir = Path(config.ACTIVE_GRAPH_DIR) / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split dir: {split_dir}")

        for graph_path in sorted(split_dir.glob("*.pkl")):
            md5 = graph_path.stem.strip().lower()
            if md5 not in label_map:
                continue

            label = int(label_map[md5])
            rows.append({
                "sample_id": md5,
                "graph_path": str(graph_path.resolve()),
                "label": label,
                "class_name": label_to_name(label),
                "origin": f"old_{split}",
                "is_new_unseen": False,
            })

    return rows


def discover_new_unseen_graphs(old_sample_ids: set[str]) -> List[dict]:
    new_dir = Path(config.CUSTOM_EVAL_NEW_MALWARE_DIR)

    if not new_dir.exists():
        if config.CUSTOM_EVAL_FAIL_IF_NEW_MALWARE_DIR_MISSING:
            raise FileNotFoundError(f"Missing CUSTOM_EVAL_NEW_MALWARE_DIR: {new_dir}")
        print(f"[WARN] New malware dir missing: {new_dir}")
        return []

    rows = []
    overlap = []

    for graph_path in sorted(new_dir.glob("*.pkl")):
        sample_id = graph_path.stem.strip().lower()

        if config.CUSTOM_EVAL_ENFORCE_NEW_UNSEEN_DISJOINT and sample_id in old_sample_ids:
            overlap.append(sample_id)
            continue

        rows.append({
            "sample_id": sample_id,
            "graph_path": str(graph_path.resolve()),
            "label": int(config.CUSTOM_EVAL_NEW_MALWARE_LABEL),
            "class_name": label_to_name(int(config.CUSTOM_EVAL_NEW_MALWARE_LABEL)),
            "origin": "new_unseen",
            "is_new_unseen": True,
        })

    if overlap:
        raise RuntimeError(
            "Overlap detected between new unseen malware and old val/test pools. "
            f"Count={len(overlap)}. Example={overlap[:10]}"
        )

    return rows


def sample_rows(rows: List[dict], n: int, name: str) -> List[dict]:
    if len(rows) < n:
        msg = f"Requested {n} {name} samples, but only found {len(rows)}."
        if name == "benign" and config.CUSTOM_EVAL_FAIL_IF_NOT_ENOUGH_BENIGN:
            raise RuntimeError(msg)
        if name == "old_malware" and config.CUSTOM_EVAL_FAIL_IF_NOT_ENOUGH_OLD_MALWARE:
            raise RuntimeError(msg)
        print(f"[WARN] {msg} Using all available.")
        return list(rows)

    rows = list(rows)
    if config.CUSTOM_EVAL_SHUFFLE_BEFORE_SAMPLE:
        rng = random.Random(config.RANDOM_SEED)
        rng.shuffle(rows)

    return rows[:n]


def build_manifest() -> pd.DataFrame:
    random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    label_map = load_metadata_labels()
    old_rows = discover_old_split_graphs(label_map)
    old_sample_ids = {r["sample_id"] for r in old_rows}
    new_rows = discover_new_unseen_graphs(old_sample_ids)

    benign_pool = [r for r in old_rows if r["label"] == 1]
    ransomware_pool = [r for r in old_rows if r["label"] == 0]
    old_malware_pool = [r for r in old_rows if r["label"] == 2]

    selected_benign = sample_rows(benign_pool, int(config.CUSTOM_EVAL_TARGET_BENIGN), "benign")
    selected_old_malware = sample_rows(old_malware_pool, int(config.CUSTOM_EVAL_TARGET_OLD_MALWARE), "old_malware")
    selected_ransomware = ransomware_pool if config.CUSTOM_EVAL_INCLUDE_ALL_RANSOMWARE else []
    selected_new_malware = new_rows if config.CUSTOM_EVAL_INCLUDE_ALL_NEW_MALWARE else []

    final_rows = selected_benign + selected_old_malware + selected_new_malware + selected_ransomware
    if not final_rows:
        raise RuntimeError("Custom evaluation manifest is empty.")

    df = pd.DataFrame(final_rows).drop_duplicates(subset=["graph_path"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    df["class_name"] = df["label"].map(label_to_name)

    config.CUSTOM_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.CUSTOM_EVAL_MANIFEST_PATH, index=False)

    print("\n[INFO] Custom manifest created:")
    print(f"       path   : {config.CUSTOM_EVAL_MANIFEST_PATH}")
    print(f"       total  : {len(df)}")
    print(f"       benign : {(df['label'] == 1).sum()}")
    print(f"       ransom : {(df['label'] == 0).sum()}")
    print(f"       other  : {(df['label'] == 2).sum()}")
    print(f"       origin : {dict(Counter(df['origin'].tolist()))}")

    return df


class CustomManifestDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        schema: Dict,
        graph_processor,
        validate_graphs: bool = True,
    ):
        self.manifest_path = Path(manifest_csv)
        self.df = pd.read_csv(self.manifest_path)
        self.schema = schema
        self.graph_processor = graph_processor
        self.validate_graphs = validate_graphs

        required = {"graph_path", "label", "sample_id"}
        missing = required - set(self.df.columns)
        if missing:
            raise KeyError(f"Manifest missing required columns: {missing}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        graph_path = Path(row["graph_path"])
        label = int(row["label"])
        sample_id = str(row["sample_id"])

        if not graph_path.exists():
            raise FileNotFoundError(f"Graph not found: {graph_path}")

        with open(graph_path, "rb") as f:
            graph = pickle.load(f)

        graph = self.graph_processor._ensure_api_timestamps(graph)

        if config.NORMALIZE_NODE_FEATURE_DIMS:
            graph = normalize_graph_node_features(graph, schema=self.schema, inplace=True)

        graph = self.graph_processor._compress_consecutive_api_runs(graph)
        graph = self.graph_processor._behavior_aware_api_cap(graph)
        graph = self.graph_processor._ensure_api_aux_tensors(graph)

        if self.validate_graphs and config.FAIL_ON_SCHEMA_MISMATCH:
            validate_graph_or_raise(
                graph,
                schema=self.schema,
                require_api_nodes=True,
                strict=config.SCHEMA_STRICT_VALIDATION,
            )

        graph.sample_id = sample_id
        graph.graph_path = str(graph_path)

        return graph, label


def _extract_timestamps(batch_data) -> dict:
    try:
        graph_timestamps = object.__getattribute__(batch_data, "graph_timestamps")
        if graph_timestamps is not None:
            return graph_timestamps
    except (AttributeError, KeyError):
        pass

    ts = {}
    try:
        for nt in batch_data.x_dict.keys():
            if hasattr(batch_data[nt], "timestamps"):
                ts[nt] = batch_data[nt].timestamps
    except (KeyError, AttributeError):
        pass

    return ts or None


def _extract_api_indices(batch_data, vocab, device):
    try:
        api_names = batch_data["api"].api_names
        return get_api_indices(api_names, vocab, device=device)
    except (AttributeError, KeyError):
        n = batch_data["api"].x.size(0)
        return torch.zeros(n, dtype=torch.long, device=device)


def _validate_batch_before_forward(batch_data, schema, api_indices=None):
    edge_attr_dict = batch_data.edge_attr_dict if hasattr(batch_data, "edge_attr_dict") else None
    batch_dict = batch_data.batch_dict if hasattr(batch_data, "batch_dict") else None

    validate_batch_or_raise(
        x_dict=batch_data.x_dict,
        edge_index_dict=batch_data.edge_index_dict,
        edge_attr_dict=edge_attr_dict,
        batch_dict=batch_dict,
        schema=schema,
        require_api=True,
        require_api_indices_if_memory=(api_indices is not None),
        api_indices=api_indices,
    )


def custom_eval_collate(batch, schema, validate_batch=False):
    if not batch:
        return None, None

    sample_ids = []
    graph_paths = []

    for graph, _label in batch:
        sample_ids.append(getattr(graph, "sample_id", ""))
        graph_paths.append(getattr(graph, "graph_path", ""))

    batched_graph, labels = hetero_collate(
        batch,
        schema=schema,
        validate_batch=validate_batch,
    )

    if batched_graph is None or labels is None:
        return batched_graph, labels

    batched_graph.sample_id = sample_ids
    batched_graph.graph_path = graph_paths
    return batched_graph, labels


@torch.no_grad()
def run_eval(model, loader, criterion, vocab, schema, device, eval_epoch=None):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    prediction_rows = []

    for batch_item in loader:
        if batch_item is None:
            continue

        if not isinstance(batch_item, (tuple, list)) or len(batch_item) != 2:
            continue

        batch_data, labels = batch_item
        if batch_data is None or labels is None:
            continue

        labels = labels.to(device)
        batch_data = batch_data.to(device)

        timestamps_dict = _extract_timestamps(batch_data)
        api_indices = _extract_api_indices(batch_data, vocab, device)

        _validate_batch_before_forward(batch_data, schema, api_indices=api_indices)

        logits = model(
            x_dict=batch_data.x_dict,
            edge_index_dict=batch_data.edge_index_dict,
            edge_attr_dict=batch_data.edge_attr_dict if hasattr(batch_data, "edge_attr_dict") else None,
            batch_dict=batch_data.batch_dict if hasattr(batch_data, "batch_dict") else None,
            timestamps_dict=timestamps_dict,
            api_indices=api_indices,
            current_epoch=eval_epoch,
        )

        loss = criterion(logits, labels)
        total_loss += float(loss.item())

        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        batch_size = labels.size(0)
        branch_weights = _get_branch_weights_from_model(model, batch_size=batch_size)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

        sample_ids = getattr(batch_data, "sample_id", None)
        graph_paths = getattr(batch_data, "graph_path", None)

        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(labels.size(0))]
        if graph_paths is None:
            graph_paths = [""] * labels.size(0)

        for i in range(labels.size(0)):
            branch_semantic = None
            branch_structural = None
            branch_temporal = None
            dominant_branch = None

            if branch_weights is not None:
                bw = branch_weights[i]
                branch_semantic = float(bw[0].item())
                branch_structural = float(bw[1].item())
                branch_temporal = float(bw[2].item())
                dominant_branch = BRANCH_NAMES[int(torch.argmax(bw).item())]

            prediction_rows.append({
                "sample_id": sample_ids[i] if isinstance(sample_ids, list) else str(sample_ids),
                "graph_path": graph_paths[i] if isinstance(graph_paths, list) else str(graph_paths),

                "true_label": int(labels[i].item()),
                "true_name": label_to_name(int(labels[i].item())),
                "pred_label": int(preds[i].item()),
                "pred_name": label_to_name(int(preds[i].item())),

                "prob_ransomware": float(probs[i, 0].item()),
                "prob_benign": float(probs[i, 1].item()),
                "prob_other_malware": float(probs[i, 2].item()),

                "branch_semantic_weight": branch_semantic,
                "branch_structural_weight": branch_structural,
                "branch_temporal_weight": branch_temporal,
                "dominant_branch": dominant_branch,

                "correct": int(preds[i].item() == labels[i].item()),
            })

    if len(all_labels) == 0:
        raise RuntimeError("No evaluation samples were processed.")

    metrics = compute_metrics(all_preds, all_labels)
    metrics["loss"] = total_loss / max(len(loader), 1)

    branch_summary = summarize_branch_usage(prediction_rows)

    return metrics, prediction_rows, branch_summary


def main():
    if not config.USE_CUSTOM_ZERO_DAY_EVAL:
        print("USE_CUSTOM_ZERO_DAY_EVAL is False in config.py. Exiting.")
        return

    config.set_seed()
    device = torch.device(config.DEVICE)
    print(f"[INFO] Device: {device}")

    schema = ensure_active_schema()
    if isinstance(schema, dict):
        active_schema = schema
    else:
        with open(config.ACTIVE_SCHEMA_PATH, "r", encoding="utf-8") as f:
            active_schema = json.load(f)

    if config.CUSTOM_EVAL_REBUILD_MANIFEST or not Path(config.CUSTOM_EVAL_MANIFEST_PATH).exists():
        manifest_df = build_manifest()
    else:
        manifest_df = pd.read_csv(config.CUSTOM_EVAL_MANIFEST_PATH)
        print(f"[INFO] Using existing manifest: {config.CUSTOM_EVAL_MANIFEST_PATH}")

    graph_processor = RansomwareGraphDataset.__new__(RansomwareGraphDataset)
    graph_processor.schema = active_schema

    dataset = CustomManifestDataset(
        manifest_csv=config.CUSTOM_EVAL_MANIFEST_PATH,
        schema=active_schema,
        graph_processor=graph_processor,
        validate_graphs=getattr(config, "CUSTOM_EVAL_VALIDATE_GRAPHS", True),
    )

    collate = partial(
        custom_eval_collate,
        schema=active_schema,
        validate_batch=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    vocab = load_vocab(config.RESULTS_DIR / "api_vocabulary.json")

    model = create_hetero_tgnn(use_memory=config.USE_MEMORY).to(device)

    checkpoint_path = Path(config.CUSTOM_EVAL_CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    missing = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print("[INFO] Loaded checkpoint with strict=False")
    if hasattr(missing, "missing_keys") and missing.missing_keys:
        print(f"[INFO] Missing keys    : {missing.missing_keys}")
    if hasattr(missing, "unexpected_keys") and missing.unexpected_keys:
        print(f"[INFO] Unexpected keys : {missing.unexpected_keys}")

    eval_epoch = int(ckpt.get("epoch", config.EPOCHS))

    print(f"[INFO] Loaded checkpoint : {checkpoint_path}")
    print(f"[INFO] Checkpoint epoch  : {eval_epoch}")
    print(f"[INFO] Eval batch size   : {config.BATCH_SIZE}")

    criterion = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)

    metrics, prediction_rows, branch_summary = run_eval(
        model=model,
        loader=loader,
        criterion=criterion,
        vocab=vocab,
        schema=active_schema,
        device=device,
        eval_epoch=eval_epoch,
    )

    print_metrics(metrics, split="Custom Zero-Day Eval")
    print_branch_usage_summary(branch_summary)

    cm = np.array(metrics["confusion_matrix"])
    tvb = compute_threat_vs_benign(cm)

    print("\nThreat vs Benign")
    print(f"  Threat Precision : {tvb['threat_precision']:.4f}")
    print(f"  Threat Recall    : {tvb['threat_recall']:.4f}")
    print(f"  Threat F1        : {tvb['threat_f1']:.4f}")
    print(f"  Benign Precision : {tvb['benign_precision']:.4f}")
    print(f"  Benign Recall    : {tvb['benign_recall']:.4f}")
    print(f"  Benign F1        : {tvb['benign_f1']:.4f}")
    print(f"  Threats Missed   : {tvb['threats_missed']}")

    pred_df = pd.DataFrame(prediction_rows)
    pred_df.to_csv(config.CUSTOM_EVAL_PREDICTIONS_PATH, index=False)

    results = {
        "manifest_path": str(config.CUSTOM_EVAL_MANIFEST_PATH),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": eval_epoch,
        "num_samples": int(len(manifest_df)),
        "eval_batch_size": int(config.BATCH_SIZE),
        "label_distribution": {
            "ransomware": int((manifest_df["label"] == 0).sum()),
            "benign": int((manifest_df["label"] == 1).sum()),
            "other_malware": int((manifest_df["label"] == 2).sum()),
        },
        "origin_distribution": {
            k: int(v) for k, v in Counter(manifest_df["origin"].tolist()).items()
        },
        "metrics_3class": metrics,
        "metrics_threat_vs_benign": tvb,
        "branch_usage_summary": branch_summary,
    }

    with open(config.CUSTOM_EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[INFO] Saved manifest    : {config.CUSTOM_EVAL_MANIFEST_PATH}")
    print(f"[INFO] Saved predictions : {config.CUSTOM_EVAL_PREDICTIONS_PATH}")
    print(f"[INFO] Saved results     : {config.CUSTOM_EVAL_RESULTS_PATH}")


if __name__ == "__main__":
    main()