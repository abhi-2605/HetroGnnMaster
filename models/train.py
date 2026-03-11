"""
scripts/train.py

Training script for Memory-Augmented Heterogeneous Temporal GNN.

Memory integration (Phase 2) adds exactly 3 changes vs previous version:
    1. get_api_indices()      → convert API names → LongTensor before forward
    2. model.memory_loss()    → add L2 reg term to task loss
    3. model.update_memory()  → EMA write after optimizer.step()

Everything else (FocalLoss, sampling, scheduler, checkpointing) unchanged.

Author: Abhishek Menon
Date:   2026-03-05
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import numpy as np
from pathlib import Path
import sys
import json
import time
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
import gc
import config
from models.tgnn_model import create_hetero_tgnn
from models.memory_utils import get_api_indices                  # ← MEMORY (1)
from graph_construction.pytorch_dataset import RansomwareGraphDataset, hetero_collate
from graph_construction.vocabulary import load_vocab             # ← MEMORY (1)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Focal Loss
# ══════════════════════════════════════════════════════════════════════════════

class FocalLoss(nn.Module):
    """
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction
        if alpha is not None:
            self.alpha = torch.tensor(alpha) if isinstance(alpha, list) else alpha
        else:
            self.alpha = None
    import gc

    

    def forward(self, inputs, targets):
        probs      = F.softmax(inputs, dim=1)
        class_mask = F.one_hot(targets, num_classes=inputs.size(1)).float()
        probs      = (probs * class_mask).sum(dim=1)
        focal_w    = (1 - probs) ** self.gamma
        ce_loss    = F.cross_entropy(inputs, targets, reduction='none')
        loss       = focal_w * ce_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            loss = self.alpha[targets] * loss

        return loss.mean() if self.reduction == 'mean' else loss.sum()


# ══════════════════════════════════════════════════════════════════════════════
#  Metrics
# ══════════════════════════════════════════════════════════════════════════════
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
        labels, predictions, average='macro', zero_division=0
    )
    w_p, w_r, w_f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    return {
        'accuracy':            acc,
        'per_class_precision': precision,
        'per_class_recall':    recall,
        'per_class_f1':        f1,
        'per_class_support':   support,
        'macro_precision':     macro_p,
        'macro_recall':        macro_r,
        'macro_f1':            macro_f1,
        'weighted_precision':  w_p,
        'weighted_recall':     w_r,
        'weighted_f1':         w_f1,
        'confusion_matrix':    confusion_matrix(labels, predictions),
    }


def print_metrics(metrics, split="Train", class_names=None):
    class_names = class_names or config.CLASS_NAMES
    print(f"{'='*70}")
    print(f"{split} Metrics")
    print(f"{'='*70}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"\nPer-Class Metrics")
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<12}")
    print(f"{'-'*70}")
    for i in range(len(metrics['per_class_precision'])):
        print(f"{class_names[i]:<20} "
              f"{metrics['per_class_precision'][i]:<12.4f} "
              f"{metrics['per_class_recall'][i]:<12.4f} "
              f"{metrics['per_class_f1'][i]:<12.4f} "
              f"{int(metrics['per_class_support'][i]):<12}")
    print(f"\nConfusion Matrix")
    print(metrics['confusion_matrix'])
    print(f"{'='*70}")


# ══════════════════════════════════════════════════════════════════════════════
#  DataLoaders
# ══════════════════════════════════════════════════════════════════════════════

def create_dataloaders(use_weighted_sampling=True):
    print("Loading datasets...")

    train_dataset = RansomwareGraphDataset(data_dir=config.ENRICHED_DIR, split='train')
    val_dataset   = RansomwareGraphDataset(data_dir=config.ENRICHED_DIR, split='val')
    test_dataset  = RansomwareGraphDataset(data_dir=config.ENRICHED_DIR, split='test')

    print(f"Train: {len(train_dataset)} graphs")
    print(f"Val:   {len(val_dataset)} graphs")
    print(f"Test:  {len(test_dataset)} graphs")

    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    print(f"\nTrain class distribution:")
    for cls in range(config.NUM_CLASSES):
        count = train_labels.count(cls)
        print(f"  {config.CLASS_NAMES[cls]}: {count} ({100*count/len(train_labels):.2f}%)")

    if use_weighted_sampling and config.USE_WEIGHTED_SAMPLING:
        print(f"\n⚡ Using weighted oversampling...")
        sample_weights = [
            config.OVERSAMPLE_WEIGHTS.get(train_dataset[i][1], 1.0)
            for i in range(len(train_dataset))
        ]
        sampler      = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)
        train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE,
            sampler=sampler, collate_fn=hetero_collate,
            num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
        )
    else:
        print(f"\n⚡ Standard sampling...")
        train_loader = DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE,
            shuffle=True, collate_fn=hetero_collate,
            num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY,
        )

    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=hetero_collate, num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
        collate_fn=hetero_collate, num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _extract_timestamps(batch_data) -> dict:
    """Safely pull timestamps from batch."""
    ts = {}
    try:
        for nt in batch_data.x_dict.keys():
            if hasattr(batch_data[nt], 'timestamps'):
                ts[nt] = batch_data[nt].timestamps
    except (KeyError, AttributeError):
        pass
    return ts or None


def _extract_api_indices(batch_data, vocab, device) -> torch.LongTensor:
    """
    Convert API name strings in the batch → LongTensor of vocab indices.  # ← MEMORY (1)
    Falls back to zeros (all <UNK>) if api_names are unavailable.
    """
    try:
        api_names = batch_data['api'].api_names   # list[str]
        return get_api_indices(api_names, vocab, device=device)
    except (AttributeError, KeyError):
        N = batch_data['api'].x.size(0)
        return torch.zeros(N, dtype=torch.long, device=device)


# ══════════════════════════════════════════════════════════════════════════════
#  train_epoch
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer, vocab, device, epoch, warmup=False,ema_model=None):
    model.train()
    oom_skip_count = 0
    total_loss    = 0.0
    total_mem_reg = 0.0
    all_preds     = []
    all_labels    = []

    # Bug 1 fix: safe defaults so pbar.set_postfix never hits NameError
    task_loss_val = 0.0
    mem_reg_val   = 0.0

    pbar = tqdm(loader, desc=f"{'WARMUP ' if warmup else ''}Epoch {epoch}")

    for batch_idx, (batch_data, labels) in enumerate(pbar):
        labels     = labels.to(device)
        batch_data = batch_data.to(device)

        batch_skipped = False

        for attempt in range(2):
            try:
                timestamps_dict = _extract_timestamps(batch_data)
                api_indices     = _extract_api_indices(batch_data, vocab, device)

                optimizer.zero_grad(set_to_none=True)

                logits = model(
                    x_dict          = batch_data.x_dict,
                    edge_index_dict = batch_data.edge_index_dict,
                    edge_attr_dict  = batch_data.edge_attr_dict,
                    batch_dict      = batch_data.batch_dict,
                    timestamps_dict = timestamps_dict,
                    api_indices     = api_indices,
                )

                task_loss = criterion(logits, labels)
                mem_reg   = model.memory_loss(lambda_reg=config.MEMORY_LAMBDA)
                loss      = task_loss + mem_reg

                loss.backward()
                if config.GRAD_CLIP > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
                optimizer.step()
                if ema_model is not None:
                        ema_model.update_parameters(model)

                # Bug 3 fix: accumulate ONLY after full forward+backward succeeds
                task_loss_val  = task_loss.item()
                mem_reg_val    = mem_reg.item()
                total_loss    += task_loss_val
                total_mem_reg += mem_reg_val

                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

                with torch.no_grad():
                    model.update_memory(
                        api_indices   = api_indices,
                        node_features = batch_data['api'].x.detach(),
                        momentum      = config.MEMORY_MOMENTUM,
                    )
                break  # IF success THEN it moves out to next batch

            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()   # Bug 2 fix: inline — no cleanup_cuda()
                gc.collect()
                oom_skip_count += 1
                if attempt == 0:
                    print(f"⚠️  OOM batch {batch_idx} — cleared cache, retrying...")
                if oom_skip_count > config.OOM_ABORT_THRESHOLD:
                    print(f"⛔ Epoch {epoch} aborted — {oom_skip_count} OOM skips, restoring checkpoint")
                    return None
                else:
                    print(f"❌  OOM batch {batch_idx} — retry failed, skipping.")
                    batch_skipped = True
                    break

        # Bug 1 + Bug 4 fix: only update postfix on successful batches
        if not batch_skipped:
            pbar.set_postfix(
                loss    = f"{task_loss_val:.4f}",
                mem_reg = f"{mem_reg_val:.1e}",
                lr      = f"{optimizer.param_groups[0]['lr']:.1e}",
            )

        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()

    avg_loss    = total_loss    / len(loader)
    avg_mem_reg = total_mem_reg / len(loader)
    metrics     = compute_metrics(all_preds, all_labels)
    metrics['loss']    = avg_loss
    metrics['mem_reg'] = avg_mem_reg

    return metrics



# ══════════════════════════════════════════════════════════════════════════════
#  validate
# ══════════════════════════════════════════════════════════════════════════════

def validate(model, loader, criterion, vocab, device):
    """Validate — memory is NOT written during eval."""
    model.eval()

    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch_data, labels in tqdm(loader, desc="Validating"):
            labels     = labels.to(device)
            batch_data = batch_data.to(device)

            timestamps_dict = _extract_timestamps(batch_data)
            api_indices     = _extract_api_indices(batch_data, vocab, device)  # ← MEMORY (1)

            logits = model(
                x_dict          = batch_data.x_dict,
                edge_index_dict = batch_data.edge_index_dict,
                edge_attr_dict  = batch_data.edge_attr_dict,
                batch_dict      = batch_data.batch_dict,
                timestamps_dict = timestamps_dict,
                api_indices     = api_indices,    # ← MEMORY (1)
            )

            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("\n" + "="*60)
    print("🔍 VALIDATION DEBUG:")
    pu, pc = np.unique(all_preds,  return_counts=True)
    lu, lc = np.unique(all_labels, return_counts=True)
    print(f"  Predicted : {dict(zip(pu, pc))}")
    print(f"  Actual    : {dict(zip(lu, lc))}")
    print("="*60 + "\n")

    metrics         = compute_metrics(all_preds, all_labels)
    metrics['loss'] = total_loss / len(loader)
    return metrics


# ══════════════════════════════════════════════════════════════════════════════
#  Main training loop
# ══════════════════════════════════════════════════════════════════════════════

def train():
    start_time = time.time()

    device = torch.device(config.DEVICE)
    print(f"\n{'='*70}\nDevice: {device}")

    # ── vocab (needed for api_indices) ───────────────────────────────────────
    vocab = load_vocab(config.RESULTS_DIR / "api_vocabulary.json")   # ← MEMORY (1)
    print(f"[Train] Vocabulary loaded: {vocab.size()} APIs")

    # ── data ─────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader = create_dataloaders(
        use_weighted_sampling=config.USE_WEIGHTED_SAMPLING
    )

    # ── model ─────────────────────────────────────────────────────────────────
    model = create_hetero_tgnn(use_memory=True).to(device)
    ema_model = None
    if config.USE_EMA:
        ema_model=  AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(config.EMA_DECAY))
    print(f"\n🚀 Model architecture:")
    print(f"  API output per layer   : {model.api_out_dim_per_layer}")
    print(f"  Resource output / layer: {model.resource_out_dim_per_layer}")
    print(f"  LSTM output            : {model.lstm_output_dim}")
    print(f"  Total embedding        : {model.total_embedding_dim}")

    # ── loss, optimiser, scheduler ───────────────────────────────────────────
    criterion = FocalLoss(alpha=config.FOCAL_ALPHA, gamma=config.FOCAL_GAMMA)
    optimizer = AdamW(model.parameters(),
                      lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    scheduler = None
    if config.USE_LR_SCHEDULER and config.LR_SCHEDULER_TYPE == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max',
            factor=config.LR_SCHEDULER_FACTOR,
            patience=config.LR_SCHEDULER_PATIENCE,
            min_lr=config.LR_MIN,
        )

    # ── resume checkpoint ─────────────────────────────────────────────────────
    checkpoint_path = config.CHECKPOINTS_DIR / 'best_model.pt'
    best_metric     = 0.0
    start_epoch     = 1

    if checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        if config.USE_EMA and ema_model is not None:
            if 'ema_state_dict' in ckpt and ckpt['ema_state_dict'] is not None:
                ema_model.load_state_dict(ckpt['ema_state_dict'])
                print("✅ EMA shadow loaded from checkpoint")
            else:
                print("ℹ️  No EMA state found — initialising shadow from current weights")

        best_metric = float(ckpt.get('best_metric', 0.0))
        start_epoch = int(ckpt.get('epoch', 0)) + 1
        print(f"🔁 Checkpoint loaded — resuming epoch {start_epoch} "
              f"| best {config.CHECKPOINT_METRIC}={best_metric:.4f}")
    else:
        print("ℹ️  No checkpoint — training from scratch.")

    # ── logging ───────────────────────────────────────────────────────────────
    log_file = config.LOGS_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    def log(msg):
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')

    patience_counter = 0
    history          = defaultdict(list)

    log(f"{'='*70}\nTRAINING START\n{'='*70}")

    # ── epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(start_epoch, config.EPOCHS + 1):
        log(f"\n{'='*70}\nEpoch {epoch}/{config.EPOCHS}\n{'='*70}")

        warmup = epoch <= config.WARMUP_EPOCHS

        # train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer,
            vocab, device, epoch, warmup=warmup,  ema_model=ema_model     # ← MEMORY: vocab passed
        )
        if train_metrics is None:
            print("⚠️  Epoch aborted — reloading weights from last good checkpoint")
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            if config.USE_EMA and ema_model is not None and 'ema_state_dict' in ckpt and ckpt['ema_state_dict'] is not None:
                ema_model.load_state_dict(ckpt['ema_state_dict'])
            continue  # skip to next epoch without validating or checkpointing

        log(f"\nTrain  loss={train_metrics['loss']:.4f}  "
            f"macro_f1={train_metrics['macro_f1']:.4f}  "
            f"mem_reg={train_metrics['mem_reg']:.6f}")  # ← MEMORY: log reg term
        

        # validate (no memory write)
        eval_model = ema_model if (config.USE_EMA and ema_model is not None) else model
        val_metrics = validate(eval_model, val_loader, criterion, vocab, device)
        log(f"Val    loss={val_metrics['loss']:.4f}  "
            f"macro_f1={val_metrics['macro_f1']:.4f}")

        # log history
        for k, v in train_metrics.items():
            if k != 'confusion_matrix':
                history[f'train_{k}'].append(
                    float(v) if not isinstance(v, np.ndarray) else v.tolist()
                )
        for k, v in val_metrics.items():
            if k != 'confusion_matrix':
                history[f'val_{k}'].append(
                    float(v) if not isinstance(v, np.ndarray) else v.tolist()
                )

        # scheduler step
        if scheduler is not None:
            scheduler.step(val_metrics[config.CHECKPOINT_METRIC])

        # checkpoint
        current = val_metrics[config.CHECKPOINT_METRIC]
        ransomware_recall = val_metrics['per_class_recall'][0]
        benign_recall     = val_metrics['per_class_recall'][1]
        recall_floors = (
            ransomware_recall >= config.MIN_RANSOMWARE_RECALL *.80 and
            benign_recall     >= config.MIN_BENIGN_RECALL     *.58
        )
        if current > best_metric and recall_floors:   # ← MEMORY: only save if recall floors are met
            best_metric      = current
            patience_counter = 0
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'ema_state_dict':       ema_model.state_dict() if ema_model is not None else None,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_metric':          best_metric,
                'val_metrics':          val_metrics,
            }, checkpoint_path)
            log(f"New best {config.CHECKPOINT_METRIC}: {current:.4f} "
        f"| Ransom recall: {ransomware_recall:.3f} "
        f"| Benign recall: {benign_recall:.3f} — saved")
            
        elif current > best_metric and not recall_floors:
            patience_counter += 1
            log(f"⚠️  macro_f1={current:.4f} but recall floors NOT met "
            f"(ransom={ransomware_recall:.3f}, benign={benign_recall:.3f}) — skipped")
        else:
            patience_counter += 1
            log(f"\n⏳ No improvement ({patience_counter}/{config.EARLY_STOPPING_PATIENCE})")
        cleanup_cuda()

        if patience_counter > config.EARLY_STOPPING_PATIENCE:
            log(f"\n🛑 Early stopping at epoch {epoch}")
            break
        
    
    # ── final test eval ───────────────────────────────────────────────────────
    log(f"\n{'='*70}\nTRAINING COMPLETE\nBest {config.CHECKPOINT_METRIC}: {best_metric:.4f}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    log(f"\n{'='*70}\nFINAL TEST EVALUATION\n{'='*70}")
    test_metrics = validate(model, test_loader, criterion, vocab, device)
    print_metrics(test_metrics, split="Test")

    history_path = config.METRICS_DIR / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    total_time = time.time() - start_time
    log(f"\n⏱️  Total time : {total_time/60:.2f} min")
    log(f"📁 Best model : {checkpoint_path}")
    log(f"📝 Log        : {log_file}")

    return model, history, test_metrics


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    config.set_seed()
    model, history, test_metrics = train()
    print("\n🎉 Training pipeline complete!")












# Batch arrives
#      │
#      ▼
# Forward pass  ──────────────────────────────────────────────┐
# (GNN + LSTM + Memory attention)                             │
#      │                                              gradient path
#      ▼                                              stays 100% intact
#   logits                                                    │
#      │                                                      │
#   task_loss + mem_reg                                       │
#      │                                                      │
#   loss.backward() ◄──────────────────────────────────────── ┘
#   optimizer.step()
#      │
#      ▼
# update_memory(node_features.detach())   ← VALUES flow in ✅
#      │                                    GRAD GRAPH stops here ✅
#      ▼                                    (we don't want/need it here)
# memory_bank slot updated with
# new real feature values ✅