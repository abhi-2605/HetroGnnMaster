"""
scripts/train.py

Training script for schema-aware TGNN with redesigned Memory v2 support.

Phase 1 schema-aware upgrades:
    1. ensure_active_schema() before dataset/model setup
    2. train/val/test loaders use ACTIVE_GRAPH_DIR + active schema
    3. validate_batch_or_raise() before model forward
    4. schema summary logged at startup

Memory v2 integration:
    1. delayed activation via current_epoch
    2. context-aware memory is handled inside tgnn_model.py
    3. confidence/frequency/novelty-aware selective write after optimizer.step()
"""

from __future__ import annotations
import importlib
import gc
import json
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
import threading
import psutil
import os
from graph_construction.pytorch_dataset import (
    SizeAwareBatchSampler,
    BucketedBatchSampler,
)
from torch.cuda.amp import GradScaler, autocast
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
_ram_log: list[dict] = []          # epoch → MB used
_ram_lock = threading.Lock()


import config
from graph_construction.pytorch_dataset import RansomwareGraphDataset, hetero_collate
from graph_construction.vocabulary import load_vocab
from models.memory_utils import (
    compose_write_strength,
    compute_confidence_write_factor,
    compute_memory_epoch_factor,
    compute_novelty_write_factor,
    compute_prediction_confidence,
    compute_frequency_write_factor,
    expand_graph_confidence_to_nodes,
    get_api_indices,
)
from models.tgnn_model import create_hetero_tgnn
from utils.schema_manager import ensure_active_schema
from utils.schema_utils import summarize_schema
from utils.schema_validator import validate_batch_or_raise


# ══════════════════════════════════════════════════════════════════════════════
#  Focal Loss
# ══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
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


# ══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════════════════════════
def _get_ram_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def _log_ram(epoch: int) -> None:
    with _ram_lock:
        _ram_log.append({"epoch": epoch, "ram_mb": _get_ram_mb()})

def _print_ram_bar(epoch: int, total_epochs: int) -> None:
    ram_mb = _get_ram_mb()
    total_mb = psutil.virtual_memory().total / 1024 / 1024
    pct = ram_mb / total_mb * 100
    filled = int(pct / 5)                       # 20-char bar
    bar = "█" * filled + "░" * (20 - filled)
    print(f"  ┌─ RAM  [{bar}] {pct:.1f}%  {ram_mb:.0f} MB / {total_mb:.0f} MB  (epoch {epoch}/{total_epochs})")

def cleanup_cuda():
    """Free Python refs + unused CUDA cached memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def compute_metrics(predictions, labels):
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    w_p, w_r, w_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    return {
        "accuracy": acc,
        "per_class_precision": precision,
        "per_class_recall": recall,
        "per_class_f1": f1,
        "per_class_support": support,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": w_p,
        "weighted_recall": w_r,
        "weighted_f1": w_f1,
        "confusion_matrix": confusion_matrix(labels, predictions),
    }


def compute_threat_vs_benign(confusion_mat: np.ndarray) -> dict:
    """
    Collapse 3-class confusion matrix into Threat (Ransom+Malware) vs Benign.
    Assumes class order: 0=Ransomware, 1=Benign, 2=Malware
    """
    # TP_threat = all threats predicted as threat (ransom or malware)
    TP = confusion_mat[0, 0] + confusion_mat[0, 2] + confusion_mat[2, 0] + confusion_mat[2, 2]
    # FN = threats predicted as benign
    FN = confusion_mat[0, 1] + confusion_mat[2, 1]
    # TN = benign predicted as benign
    TN = confusion_mat[1, 1]
    # FP = benign predicted as threat
    FP = confusion_mat[1, 0] + confusion_mat[1, 2]

    threat_recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    threat_precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    threat_f1        = (
        2 * threat_precision * threat_recall / (threat_precision + threat_recall)
        if (threat_precision + threat_recall) > 0 else 0.0
    )
    benign_recall    = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    benign_precision = TN / (TN + FN) if (TN + FN) > 0 else 0.0
    benign_f1        = (
        2 * benign_precision * benign_recall / (benign_precision + benign_recall)
        if (benign_precision + benign_recall) > 0 else 0.0
    )
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    return {
        "TP": int(TP), "FN": int(FN), "TN": int(TN), "FP": int(FP),
        "threat_recall": threat_recall, "threat_precision": threat_precision, "threat_f1": threat_f1,
        "benign_recall": benign_recall, "benign_precision": benign_precision, "benign_f1": benign_f1,
        "accuracy_2class": accuracy,
        "threats_missed": int(FN),   # most important number
    }



def print_metrics(metrics, split="Train", class_names=None):
    class_names = class_names or config.CLASS_NAMES
    print(f"{'=' * 70}")
    print(f"{split} Metrics")
    print(f"{'=' * 70}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print("\nPer-Class Metrics")
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<12}")
    print(f"{'-' * 70}")
    for i in range(len(metrics["per_class_precision"])):
        print(
            f"{class_names[i]:<20} "
            f"{metrics['per_class_precision'][i]:<12.4f} "
            f"{metrics['per_class_recall'][i]:<12.4f} "
            f"{metrics['per_class_f1'][i]:<12.4f} "
            f"{int(metrics['per_class_support'][i]):<12}"
        )
    print("\nConfusion Matrix")
    print(metrics["confusion_matrix"])
    print(f"{'=' * 70}")


# ══════════════════════════════════════════════════════════════════════════════
#  DataLoaders
# ══════════════════════════════════════════════════════════════════════════════

def create_dataloaders(schema, use_weighted_sampling=True):
    print("Loading datasets...")
    print(f"Active graph source: {config.ACTIVE_GRAPH_SOURCE}")
    print(f"Active graph dir   : {config.ACTIVE_GRAPH_DIR}")

    train_dataset = RansomwareGraphDataset(
        data_dir=str(config.DATA_DIR),
        split="train",
        cache_dir=str(config.ACTIVE_GRAPH_DIR),
        schema=schema,
        validate_graphs=True,
    )
    val_dataset = RansomwareGraphDataset(
        data_dir=str(config.DATA_DIR),
        split="val",
        cache_dir=str(config.ACTIVE_GRAPH_DIR),
        schema=schema,
        validate_graphs=True,
    )
    test_dataset = RansomwareGraphDataset(
        data_dir=str(config.DATA_DIR),
        split="test",
        cache_dir=str(config.ACTIVE_GRAPH_DIR),
        schema=schema,
        validate_graphs=True,
    )

    print(f"Train: {len(train_dataset)} graphs")
    print(f"Val:   {len(val_dataset)} graphs")
    print(f"Test:  {len(test_dataset)} graphs")

    train_labels =  [
    train_dataset.labels.get(graph_file.stem, 2)
    for graph_file in train_dataset.graph_files
    ]
    print("\nTrain class distribution:")
    for cls in range(config.NUM_CLASSES):
        count = train_labels.count(cls)
        pct = 100 * count / len(train_labels) if train_labels else 0.0
        print(f"  {config.CLASS_NAMES[cls]}: {count} ({pct:.2f}%)")

    collate = lambda batch: hetero_collate(batch, schema=schema, validate_batch=False)

    # ── build weights if weighted sampling is enabled ─────────────────────────
    sample_weights = None
    if use_weighted_sampling and config.USE_WEIGHTED_SAMPLING:
        sample_weights = [
            config.OVERSAMPLE_WEIGHTS.get(
                train_dataset.labels.get(graph_file.stem, 2), 1.0
            )
            for graph_file in train_dataset.graph_files
        ]

    # ── choose sampler from config ────────────────────────────────────────────
    if getattr(config, "ENABLE_BUCKETED_BATCH_SAMPLER", False):
        print("\n⚡ Using weighted oversampling + bucketed batching..." if sample_weights is not None
              else "\n⚡ Using bucketed batching...")

        train_batch_sampler = BucketedBatchSampler(
        dataset=train_dataset,
        bucket_rules=getattr(config, "BUCKETED_BATCH_RULES", None),
        shuffle=getattr(config, "BUCKETED_BATCH_SHUFFLE", True),
        weights=sample_weights,
        verbose=getattr(config, "BUCKETED_BATCH_VERBOSE", True),
    )
    else:
        print("\n⚡ Using weighted oversampling + size-aware batching..." if sample_weights is not None
              else "\n⚡ Using size-aware batching...")

        train_batch_sampler = BucketedBatchSampler(
            dataset=train_dataset,
            bucket_rules=getattr(config, "BUCKETED_BATCH_RULES", None),
            shuffle=getattr(config, "BUCKETED_BATCH_SHUFFLE", True),
            weights=sample_weights,
            verbose=getattr(config, "BUCKETED_BATCH_VERBOSE", True),
        )

    # ── single size-aware sampler handles both cases ──────────────────────────
    # size_sampler = SizeAwareBatchSampler(
    #     train_dataset,
    #     batch_size=config.BATCH_SIZE,
    #     shuffle=True,
    #     weights=sample_weights,   # None → uniform, list → weighted
    # )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,   # replaces batch_size + sampler/shuffle
        collate_fn=collate,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader



# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _extract_timestamps(batch_data) -> dict:
    """Safely pull timestamps from batch."""
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


def _extract_api_indices(batch_data, vocab, device) -> torch.LongTensor:
    """
    Convert API name strings in the batch → LongTensor of vocab indices.
    Falls back to zeros (all <UNK>) if api_names are unavailable.
    """
    try:
        api_names = batch_data["api"].api_names
        return get_api_indices(api_names, vocab, device=device)
    except (AttributeError, KeyError):
        n = batch_data["api"].x.size(0)
        return torch.zeros(n, dtype=torch.long, device=device)


def _validate_batch_before_forward(batch_data, schema, api_indices=None):
    """
    Validate model-ready batch dictionaries before forward pass.
    """
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


def _build_memory_write_strength(model, logits, batch_data, api_indices, epoch):
    """
    Compose selective memory write strength using:
    - delayed activation factor
    - confidence-aware scaling
    - frequency-aware scaling
    - novelty-aware scaling

    Returns:
        write_strength: [N_api] or None
    """
    if not getattr(config, "USE_MEMORY", False):
        return None

    if not getattr(model, "memory_enabled_for_temporal_type", False):
        return None

    if getattr(model, "memory", None) is None:
        return None

    cache = getattr(model, "_memory_cache", None)
    if not cache or not cache.get("memory_enabled", False):
        return None

    if api_indices is None or api_indices.numel() == 0:
        return None

    device = api_indices.device
    num_nodes = api_indices.size(0)

    epoch_factor_scalar = compute_memory_epoch_factor(
        epoch=epoch,
        warmup_epochs=getattr(config, "MEMORY_WARMUP_EPOCHS", 5),
    )
    if epoch_factor_scalar <= 0.0:
        return None

    epoch_factor = torch.full(
        (num_nodes,),
        fill_value=float(epoch_factor_scalar),
        dtype=torch.float32,
        device=device,
    )

    confidence_factor = None
    if getattr(config, "USE_CONFIDENCE_AWARE_WRITE", True):
        graph_conf = compute_prediction_confidence(logits.detach())  # [B]
        primary_type = model.primary_temporal_type
        node_batch = batch_data.batch_dict[primary_type]
        node_conf = expand_graph_confidence_to_nodes(graph_conf, node_batch)
        confidence_factor = compute_confidence_write_factor(
            confidence=node_conf,
            min_confidence=getattr(config, "MEMORY_WRITE_MIN_CONFIDENCE", 0.60),
        )

    frequency_factor = None
    if getattr(config, "USE_FREQUENCY_AWARE_WRITE", True):
        freq_table = getattr(model.memory.vocab, "get_frequency_table", lambda: {})()
        frequency_factor = compute_frequency_write_factor(
            api_indices=api_indices,
            freq_table=freq_table,
            alpha=getattr(config, "MEMORY_FREQ_ALPHA", 0.5),
            eps=getattr(config, "MEMORY_FREQ_EPS", 1e-6),
            common_token_cap=getattr(config, "MEMORY_COMMON_TOKEN_WRITE_CAP", 0.25),
        )

    novelty_factor = None
    if getattr(config, "USE_NOVELTY_AWARE_WRITE", True):
        write_node_features = cache.get("write_node_features", None)
        write_context = cache.get("write_context", None)
        memory_read = cache.get("memory_read", None)

        if (
            write_node_features is not None
            and memory_read is not None
            and hasattr(model.memory, "_build_write_candidate")
        ):
            write_candidate = model.memory._build_write_candidate(
                api_indices=api_indices,
                node_features=write_node_features,
                write_context=write_context,
            ).detach()

            novelty_factor = compute_novelty_write_factor(
                write_candidate=write_candidate,
                memory_reference=memory_read,
                temperature=getattr(config, "MEMORY_NOVELTY_TEMPERATURE", 1.0),
                eps=getattr(config, "MEMORY_NOVELTY_EPS", 1e-6),
            )

    write_strength = compose_write_strength(
        num_items=num_nodes,
        device=device,
        base_strength=getattr(config, "MEMORY_WRITE_BASE_STRENGTH", 1.0),
        confidence_factor=confidence_factor,
        frequency_factor=frequency_factor,
        novelty_factor=novelty_factor,
        epoch_factor=epoch_factor,
        min_strength=getattr(config, "MEMORY_WRITE_MIN_STRENGTH", 0.05),
        max_strength=getattr(config, "MEMORY_WRITE_MAX_STRENGTH", 1.0),
    )
    return write_strength


# ══════════════════════════════════════════════════════════════════════════════
#  train_epoch
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer, vocab, schema, device, epoch, warmup=False, scaler=None):
    model.train()

    total_loss = 0.0
    total_mem_reg = 0.0
    processed_batches = 0

    all_preds = []
    all_labels = []

    task_loss_val = 0.0
    mem_reg_val = 0.0

    pbar = tqdm(loader, desc=f"{'WARMUP ' if warmup else ''}Epoch {epoch}")

    for batch_idx, batch_item in enumerate(pbar):
        # Guard against collate returning None/empty batch
        if batch_item is None:
            print(f"⚠️  Empty batch object at batch {batch_idx} — skipping.")
            continue

        if not isinstance(batch_item, (tuple, list)) or len(batch_item) != 2:
            print(f"⚠️  Malformed batch at batch {batch_idx} — skipping.")
            continue

        batch_data, labels = batch_item

        if batch_data is None or labels is None:
            print(f"⚠️  Empty batch at batch {batch_idx} — skipping.")
            continue

        labels = labels.to(device)
        batch_data = batch_data.to(device)

        batch_skipped = False

        for attempt in range(2):
            try:
                timestamps_dict = _extract_timestamps(batch_data)
                api_indices = _extract_api_indices(batch_data, vocab, device)

                _validate_batch_before_forward(batch_data, schema, api_indices=api_indices)

                optimizer.zero_grad(set_to_none=True)

                use_amp = False
                with torch.amp.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    logits = model(
                        x_dict=batch_data.x_dict,
                        edge_index_dict=batch_data.edge_index_dict,
                        edge_attr_dict=batch_data.edge_attr_dict,
                        batch_dict=batch_data.batch_dict,
                        timestamps_dict=timestamps_dict,
                        api_indices=api_indices,
                        current_epoch=epoch,
                    )

                # Compute losses OUTSIDE autocast for numerical stability
                logits = logits.float()

                if not torch.isfinite(logits).all():
                    print(f"❌ Non-finite logits at batch {batch_idx} — skipping batch.")
                    batch_skipped = True
                    break

                task_loss = criterion(logits, labels)
                mem_reg = model.memory_loss(lambda_reg=config.MEMORY_LAMBDA)
                loss = task_loss + mem_reg

                if not torch.isfinite(task_loss):
                    print(f"❌ Non-finite task_loss at batch {batch_idx} — skipping batch.")
                    batch_skipped = True
                    break

                if not torch.isfinite(mem_reg):
                    print(f"❌ Non-finite mem_reg at batch {batch_idx} — skipping batch.")
                    batch_skipped = True
                    break

                if not torch.isfinite(loss):
                    print(f"❌ Non-finite total loss at batch {batch_idx} — skipping batch.")
                    batch_skipped = True
                    break

                if use_amp:
                    if scaler is None:
                        raise RuntimeError("AMP is enabled but scaler is None.")

                    scaler.scale(loss).backward()

                    if config.GRAD_CLIP > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)

                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()

                    if config.GRAD_CLIP > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)

                    optimizer.step()

                write_strength = None
                if getattr(config, "USE_MEMORY", False):
                    write_strength = _build_memory_write_strength(
                        model=model,
                        logits=logits,
                        batch_data=batch_data,
                        api_indices=api_indices,
                        epoch=epoch,
                    )

                    model.update_memory(
                        api_indices=api_indices,
                        node_features=batch_data[model.primary_temporal_type].x.detach(),
                        momentum=config.MEMORY_MOMENTUM,
                        write_strength=write_strength,
                        current_epoch=epoch,
                    )

                task_loss_val = float(task_loss.item())
                mem_reg_val = float(mem_reg.item())

                total_loss += task_loss_val
                total_mem_reg += mem_reg_val
                processed_batches += 1

                preds = logits.argmax(dim=1).detach().cpu().numpy()
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.detach().cpu().numpy().tolist())

                break

            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                if attempt == 0:
                    print(f"⚠️  OOM batch {batch_idx} — cleared cache, retrying...")
                else:
                    print(f"❌  OOM batch {batch_idx} — retry failed, skipping.")
                    batch_skipped = True
                    break

            except Exception as e:
                optimizer.zero_grad(set_to_none=True)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                print(f"❌ Exception in batch {batch_idx}: {type(e).__name__}: {e}")
                batch_skipped = True
                break

        if not batch_skipped:
            pbar.set_postfix(
                loss=f"{task_loss_val:.4f}",
                mem_reg=f"{mem_reg_val:.1e}",
                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
            )

        if batch_idx % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_loss = total_loss / max(processed_batches, 1)
    avg_mem_reg = total_mem_reg / max(processed_batches, 1)

    # Safe fallback if every batch was skipped
    if len(all_preds) == 0 or len(all_labels) == 0:
        return {
            "accuracy": 0.0,
            "per_class_precision": np.zeros(config.NUM_CLASSES, dtype=float),
            "per_class_recall": np.zeros(config.NUM_CLASSES, dtype=float),
            "per_class_f1": np.zeros(config.NUM_CLASSES, dtype=float),
            "per_class_support": np.zeros(config.NUM_CLASSES, dtype=float),
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "weighted_precision": 0.0,
            "weighted_recall": 0.0,
            "weighted_f1": 0.0,
            "confusion_matrix": np.zeros((config.NUM_CLASSES, config.NUM_CLASSES), dtype=int),
            "loss": avg_loss,
            "mem_reg": avg_mem_reg,
        }

    metrics = compute_metrics(all_preds, all_labels)
    metrics["loss"] = avg_loss
    metrics["mem_reg"] = avg_mem_reg
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  validate
# ══════════════════════════════════════════════════════════════════════════════

def validate(model, loader, criterion, vocab, schema, device, epoch=None):
    """Validate — memory is NOT written during eval."""
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_data, labels in tqdm(loader, desc="Validating"):
            labels = labels.to(device)
            batch_data = batch_data.to(device)

            timestamps_dict = _extract_timestamps(batch_data)
            api_indices = _extract_api_indices(batch_data, vocab, device)

            _validate_batch_before_forward(batch_data, schema, api_indices=api_indices)

            logits = model(
                x_dict=batch_data.x_dict,
                edge_index_dict=batch_data.edge_index_dict,
                edge_attr_dict=batch_data.edge_attr_dict,
                batch_dict=batch_data.batch_dict,
                timestamps_dict=timestamps_dict,
                api_indices=api_indices,
                current_epoch=epoch,
            )

            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("\n" + "=" * 60)
    print("🔍 VALIDATION DEBUG:")
    pu, pc = np.unique(all_preds, return_counts=True)
    lu, lc = np.unique(all_labels, return_counts=True)
    print(f"  Predicted : {dict(zip(pu, pc))}")
    print(f"  Actual    : {dict(zip(lu, lc))}")
    print("=" * 60 + "\n")

    metrics = compute_metrics(all_preds, all_labels)
    metrics["loss"] = total_loss / len(loader)
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  Main training loop
# ══════════════════════════════════════════════════════════════════════════════

def train():
    start_time = time.time()

    device = torch.device(config.DEVICE)
    print(f"\n{'=' * 70}\nDevice: {device}")

    # ── schema ────────────────────────────────────────────────────────────────
    schema = ensure_active_schema()
    schema_summary = summarize_schema(schema)

    print(f"\n{'=' * 70}")
    print("ACTIVE SCHEMA")
    print(f"{'=' * 70}")
    print(f"Source             : {schema_summary['source_name']}")
    print(f"Graph root         : {schema_summary['graph_root']}")
    print(f"Node types         : {schema_summary['node_types']}")
    print(f"Edge types         : {schema_summary['edge_types']}")
    print(f"Canonical node dims: {schema_summary['canonical_node_dims']}")
    print(f"Non-empty edges    : {schema_summary['nonempty_edge_types']}")
    print(f"Empty edges        : {schema_summary['empty_edge_types']}")

    # ── vocab ─────────────────────────────────────────────────────────────────
    vocab = load_vocab(config.RESULTS_DIR / "api_vocabulary.json")
    print(f"[Train] Vocabulary loaded: {vocab.size()} APIs")
    

    # ── data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = create_dataloaders(
        schema=schema,
        use_weighted_sampling=config.USE_WEIGHTED_SAMPLING,
    )

    # ── model ─────────────────────────────────────────────────────────────────
    model = create_hetero_tgnn(use_memory=config.USE_MEMORY).to(device)

    print("\n🚀 Model architecture:")
    print(f"  Primary temporal type  : {model.primary_temporal_type}")
    print(f"  Primary output / layer : {model.primary_out_dim_per_layer}")
    print(f"  Other output / layer   : {model.other_out_dim_per_layer}")
    print(f"  LSTM output            : {model.lstm_output_dim}")
    print(f"  Total embedding        : {model.total_embedding_dim}")

    # ── loss, optimiser, scheduler ───────────────────────────────────────────
    criterion = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    scheduler = None
    if config.USE_LR_SCHEDULER and config.LR_SCHEDULER_TYPE == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE,
            min_lr=config.LR_MIN,
        )

    # ── resume checkpoint ─────────────────────────────────────────────────────
    checkpoint_path = config.CHECKPOINTS_DIR / "best_model.pt"
    best_metric = 0.0
    start_epoch = 1
    use_amp = False
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if config.RESUME_FROM_CHECKPOINT and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("scaler_state_dict") is not None:   
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        best_metric = float(ckpt.get("best_metric", 0.0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(
            f"🔁 Checkpoint loaded — resuming epoch {start_epoch} "
            f"| best {config.CHECKPOINT_METRIC}={best_metric:.4f}"
        )
    else:
        print("ℹ️  No checkpoint — training from scratch.")

    # ── logging ───────────────────────────────────────────────────────────────
    log_file = config.LOGS_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    def log(msg):
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    patience_counter = 0
    history = defaultdict(list)

    active_schema_path = getattr(config, "SCHEMA_PATH", None)

    log(f"{'=' * 70}\nTRAINING START\n{'=' * 70}")
    log(f"Graph source: {config.ACTIVE_GRAPH_SOURCE}")
    log(f"Graph dir   : {config.ACTIVE_GRAPH_DIR}")
    if active_schema_path is not None:
        log(f"Schema path : {active_schema_path}")

    # ── epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, config.EPOCHS + 1):
        importlib.reload(config)  # hot-reload utils for dynamic config changes
        log(f"\n{'=' * 70}\nEpoch {epoch}/{config.EPOCHS}\n{'=' * 70}")

        warmup = epoch <= config.WARMUP_EPOCHS

        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            vocab=vocab,
            schema=schema,
            device=device,
            epoch=epoch,
            warmup=warmup,
            scaler=scaler
        )
        _log_ram(epoch)
        _print_ram_bar(epoch, config.EPOCHS)


        ransom_r = train_metrics["per_class_recall"][0]
        benign_r = train_metrics["per_class_recall"][1]
        malware_r = train_metrics["per_class_recall"][2]
        ransom_p = train_metrics["per_class_precision"][0]
        benign_p = train_metrics["per_class_precision"][1]
        malware_p = train_metrics["per_class_precision"][2]

        log(
            f"\nTrain  loss={train_metrics['loss']:.4f}  "
            f"macro_f1={train_metrics['macro_f1']:.4f}  "
            f"mem_reg={train_metrics['mem_reg']:.6f}  "
            f"Ransom P={ransom_p:.3f} R={ransom_r:.3f}  "
            f"Benign P={benign_p:.3f} R={benign_r:.3f}  "
            f"Malware P={malware_p:.3f} R={malware_r:.3f}"
        )

        val_metrics = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            vocab=vocab,
            schema=schema,
            device=device,
            epoch=epoch,
        )

        ransom_v_r = val_metrics["per_class_recall"][0]
        benign_v_r = val_metrics["per_class_recall"][1]
        malware_v_r = val_metrics["per_class_recall"][2]
        ransom_v_p = val_metrics["per_class_precision"][0]
        benign_v_p = val_metrics["per_class_precision"][1]
        malware_v_p = val_metrics["per_class_precision"][2]

        log(
            f"Val    loss={val_metrics['loss']:.4f}  "
            f"macro_f1={val_metrics['macro_f1']:.4f}  "
            f"Ransom P={ransom_v_p:.3f} R={ransom_v_r:.3f}  "
            f"Benign P={benign_v_p:.3f} R={benign_v_r:.3f}  "
            f"Malware P={malware_v_p:.3f} R={malware_v_r:.3f}"
        )

        # ── Threat vs Benign 2-class report ──────────────────────────────────
        tvb = compute_threat_vs_benign(val_metrics["confusion_matrix"])
        log(
            f"  [Threat vs Benign]  "
            f"Threat P={tvb['threat_precision']:.3f} R={tvb['threat_recall']:.3f} F1={tvb['threat_f1']:.3f}  |  "
            f"Benign P={tvb['benign_precision']:.3f} R={tvb['benign_recall']:.3f} F1={tvb['benign_f1']:.3f}  |  "
            f"Acc={tvb['accuracy_2class']:.4f}  Threats missed={tvb['threats_missed']}"
        )

        for k, v in train_metrics.items():
            if k != "confusion_matrix":
                history[f"train_{k}"].append(float(v) if not isinstance(v, np.ndarray) else v.tolist())
        for k, v in val_metrics.items():
            if k != "confusion_matrix":
                history[f"val_{k}"].append(float(v) if not isinstance(v, np.ndarray) else v.tolist())

        if scheduler is not None:
            scheduler.step(val_metrics[config.CHECKPOINT_METRIC])

        current = val_metrics[config.CHECKPOINT_METRIC]
        ransomware_recall = val_metrics["per_class_recall"][0]
        benign_recall = val_metrics["per_class_recall"][1]

        recall_floors = (
            ransomware_recall >= config.MIN_RANSOMWARE_RECALL * 0.64
            and benign_recall >= config.MIN_BENIGN_RECALL * 0.58
        )

        if current > best_metric and recall_floors:
            best_metric = current
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "best_metric": best_metric,
                    "val_metrics": val_metrics,
                    "graph_source": config.ACTIVE_GRAPH_SOURCE,
                    "scaler_state_dict": scaler.state_dict(),
                    "schema_path": str(active_schema_path) if active_schema_path is not None else None,
                },
                checkpoint_path,
            )
            log(
                f"New best {config.CHECKPOINT_METRIC}: {current:.4f} "
                f"| Ransom recall: {ransomware_recall:.3f} "
                f"| Benign recall: {benign_recall:.3f} — saved"
            )

        elif current > best_metric and not recall_floors:
            patience_counter += 1
            log(
                f"⚠️  macro_f1={current:.4f} but recall floors NOT met "
                f"(ransom={ransomware_recall:.3f}, benign={benign_recall:.3f}) — skipped"
            )
        else:
            patience_counter += 1
            log(f"\n⏳ No improvement ({patience_counter}/{config.EARLY_STOPPING_PATIENCE})")

        cleanup_cuda()

        if patience_counter > config.EARLY_STOPPING_PATIENCE:
            log(f"\n🛑 Early stopping at epoch {epoch}")
            break

    # ── final test eval ───────────────────────────────────────────────────────
    log(f"\n{'=' * 70}\nTRAINING COMPLETE\nBest {config.CHECKPOINT_METRIC}: {best_metric:.4f}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    log(f"\n{'=' * 70}\nFINAL TEST EVALUATION\n{'=' * 70}")
    test_metrics = validate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        vocab=vocab,
        schema=schema,
        device=device,
        epoch=config.EPOCHS,
    )
    print_metrics(test_metrics, split="Test")

    history_path = config.METRICS_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - start_time
    log(f"\n⏱️  Total time : {total_time / 60:.2f} min")
    log(f"📁 Best model : {checkpoint_path}")
    log(f"📝 Log        : {log_file}")

    return model, history, test_metrics


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    config.set_seed()
    model, history, test_metrics = train()
    print("\n🎉 Training pipeline complete!")