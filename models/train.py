"""
Training script for Heterogeneous Temporal GNN (Ransomware Detection).

Implements FULL POWER training strategy:
1. Focal Loss with boosted class weights
2. Weighted oversampling for class balance
3. Multi-metric evaluation (per-class F1, macro F1, recall, precision)
4. Early stopping on macro F1
5. Learning rate scheduler (ReduceLROnPlateau)
6. Warmup epochs
7. Two-stage training (optional)
8. Gradient clipping
9. Comprehensive logging and checkpointing

Author: Your Name
Date: 2026-02-12
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
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

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import config
from models.tgnn_model import create_hetero_tgnn
from graph_construction.pytorch_dataset import RansomwareGraphDataset, hetero_collate

# Sklearn metrics
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


# ============================================
# FOCAL LOSS IMPLEMENTATION
# ============================================

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha: Class weights [C] (e.g., [0.5, 0.35, 0.15])
        gamma: Focusing parameter (2.0 recommended)
        reduction: 'mean' or 'sum'
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
        if alpha is not None:
            if isinstance(alpha, list):
                self.alpha = torch.tensor(alpha)
            else:
                self.alpha = alpha
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: [batch_size, num_classes] logits
            targets: [batch_size] labels (0, 1, 2)
        """
        # Get probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Get class probabilities
        class_mask = F.one_hot(targets, num_classes=inputs.size(1))
        probs = (probs * class_mask).sum(dim=1)  # [batch_size]
        
        # Compute focal term: (1 - p_t)^gamma
        focal_weight = (1 - probs) ** self.gamma
        
        # Compute cross entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        # Apply class weights (alpha)
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================
# METRIC COMPUTATION
# ============================================

def compute_metrics(predictions, labels, num_classes=3):
    """
    Compute comprehensive metrics.
    
    Returns dict with:
    - accuracy
    - per_class_precision, per_class_recall, per_class_f1
    - macro_precision, macro_recall, macro_f1
    - weighted_precision, weighted_recall, weighted_f1
    - confusion_matrix
    """
    # Sklearn metrics
    acc = accuracy_score(labels, predictions)
    
    # Per-class and macro/weighted metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )
    
    cm = confusion_matrix(labels, predictions)
    
    metrics = {
        'accuracy': acc,
        # Per-class
        'per_class_precision': precision,
        'per_class_recall': recall,
        'per_class_f1': f1,
        'per_class_support': support,
        # Macro (equal weight to all classes)
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        # Weighted by class frequency
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        # Confusion matrix
        'confusion_matrix': cm,
    }
    
    return metrics


def print_metrics(metrics, split="Train", class_names=None):
    """Pretty print metrics."""
    if class_names is None:
        class_names = config.CLASS_NAMES
    
    print(f"{'='*70}")
    print(f"{split} Metrics")
    print(f"{'='*70}")
    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"\n{'Per-Class Metrics'}")
    print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<12}")
    print(f"{'-'*70}")
    for i in range(len(metrics['per_class_precision'])):
        print(f"{class_names[i]:<20} "
              f"{metrics['per_class_precision'][i]:<12.4f} "
              f"{metrics['per_class_recall'][i]:<12.4f} "
              f"{metrics['per_class_f1'][i]:<12.4f} "
              f"{int(metrics['per_class_support'][i]):<12}")
    
    print(f"\n{'Confusion Matrix'}")
    print(metrics['confusion_matrix'])
    print(f"{'='*70}")


# ============================================
# DATA LOADING
# ============================================

def create_dataloaders(use_weighted_sampling=True):
    """
    Create train/val/test dataloaders.
    
    Args:
        use_weighted_sampling: If True, use WeightedRandomSampler for training
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print("Loading datasets...")
    
    # Load datasets
    train_dataset = RansomwareGraphDataset(
        data_dir=config.ENRICHED_DIR,
        split='train'
    )
    
    val_dataset = RansomwareGraphDataset(
        data_dir=config.ENRICHED_DIR,
        split='val'
    )
    
    test_dataset = RansomwareGraphDataset(
        data_dir=config.ENRICHED_DIR,
        split='test'
    )
    
    print(f"Train: {len(train_dataset)} graphs")
    print(f"Val:   {len(val_dataset)} graphs")
    print(f"Test:  {len(test_dataset)} graphs")
    
    # Count class distribution
    train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
    print(f"\n{'Train class distribution'}")
    for cls in range(config.NUM_CLASSES):
        count = train_labels.count(cls)
        print(f"  {config.CLASS_NAMES[cls]}: {count} ({100 * count / len(train_labels):.2f}%)")
    
    # Create samplers
    if use_weighted_sampling and config.USE_WEIGHTED_SAMPLING:
        print(f"\n⚡ Using weighted oversampling...")
        sample_weights = []
        for idx in range(len(train_dataset)):
            label = train_dataset[idx][1]
            weight = config.OVERSAMPLE_WEIGHTS.get(label, 1.0)
            sample_weights.append(weight)
        
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(train_dataset),
            replacement=True
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            sampler=sampler,
            collate_fn=hetero_collate,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )
    else:
        print(f"\n⚡ Using standard sampling (no oversampling)...")
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            collate_fn=hetero_collate,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )
    
    # Val and test (no sampling)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=hetero_collate,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=hetero_collate,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    return train_loader, val_loader, test_loader


# ============================================
# TRAINING & VALIDATION
# ============================================

def train_epoch(model, loader, criterion, optimizer, device, epoch, warmup=False):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, desc=f"{'WARMUP ' if warmup else ''}Epoch {epoch}")
    
    for batch_idx, (batch_data, labels) in enumerate(pbar):
        # Move to device
        labels = labels.to(device)
        batch_data = batch_data.to(device)

        # ✅ FIXED: Safe timestamp extraction using x_dict keys
        timestamps_dict = None
        try:
            timestamps_dict = {}
            for node_type in batch_data.x_dict.keys():
                if hasattr(batch_data[node_type], 'timestamps'):
                    timestamps_dict[node_type] = batch_data[node_type].timestamps
            
            if not timestamps_dict:
                timestamps_dict = None
        except (KeyError, AttributeError):
            timestamps_dict = None
        
        # Forward
        optimizer.zero_grad()
        logits = model(
            x_dict=batch_data.x_dict,
            edge_index_dict=batch_data.edge_index_dict,
            edge_attr_dict=batch_data.edge_attr_dict,
            batch_dict=batch_data.batch_dict,
            timestamps_dict=timestamps_dict
        )
        
        # Loss
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        if config.GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        
        optimizer.step()
        
        # Update progress bar
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Compute metrics
    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = avg_loss
    
    return metrics

def validate(model, loader, criterion, device):
    """Validate."""
    model.eval()
    
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data, labels in tqdm(loader, desc="Validating"):
            labels = labels.to(device)
            batch_data = batch_data.to(device)
            
            # ✅ FIXED: Safe timestamp extraction using x_dict keys
            timestamps_dict = None
            try:
                timestamps_dict = {}
                for node_type in batch_data.x_dict.keys():
                    if hasattr(batch_data[node_type], 'timestamps'):
                        timestamps_dict[node_type] = batch_data[node_type].timestamps
                
                if not timestamps_dict:
                    timestamps_dict = None
            except (KeyError, AttributeError):
                timestamps_dict = None
            
            # Forward
            logits = model(
                x_dict=batch_data.x_dict,
                edge_index_dict=batch_data.edge_index_dict,
                edge_attr_dict=batch_data.edge_attr_dict,
                batch_dict=batch_data.batch_dict,
                timestamps_dict=timestamps_dict
            )
            
            # Loss
            loss = criterion(logits, labels)
            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 🔍 DEBUG: Print prediction distribution
    print("\n" + "="*60)
    print("🔍 VALIDATION DEBUG:")
    pred_unique, pred_counts = np.unique(all_preds, return_counts=True)
    label_unique, label_counts = np.unique(all_labels, return_counts=True)
    print(f"Predicted: {dict(zip(pred_unique, pred_counts))}")
    print(f"Actual:    {dict(zip(label_unique, label_counts))}")
    print("="*60 + "\n")
    
    # Compute metrics
    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = avg_loss
    
    return metrics





# ============================================
# MAIN TRAINING LOOP
# ============================================

def train():
    """Main training function."""
    
    start_time = time.time()
    
    # Device
    device = torch.device(config.DEVICE)
    print(f"\n{'='*70}")
    print(f"Device: {device}")
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        use_weighted_sampling=config.USE_WEIGHTED_SAMPLING
    )
    
    # Create model
    model = create_hetero_tgnn()
    model = model.to(device)
    
    print(f"\n🚀 Model architecture:")
    print(f" API output per layer: {model.api_out_dim_per_layer}")
    print(f" Resource output per layer: {model.resource_out_dim_per_layer}")
    print(f" LSTM output: {model.lstm_output_dim}")
    print(f" Total embedding: {model.total_embedding_dim}")
    print(f" Classifier hidden: {model.classifier_hidden_dim}")
    
    # Loss function (Focal Loss)
    criterion = FocalLoss(
        alpha=config.FOCAL_ALPHA,
        gamma=config.FOCAL_GAMMA
    )
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    if config.USE_LR_SCHEDULER:
        print(f" Using {config.LR_SCHEDULER_TYPE} scheduler")
        if config.LR_SCHEDULER_TYPE == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=config.LR_SCHEDULER_FACTOR,
                patience=config.LR_SCHEDULER_PATIENCE,
                min_lr=config.LR_MIN,
            )
        else:
            scheduler = None
    else:
        scheduler = None
    
    # Training state
    best_metric = 0.0
    patience_counter = 0
    history = defaultdict(list)
    
    # Training log file
    log_file = config.LOGS_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    def log(msg):
        """Log to console and file."""
        print(msg)
        with open(log_file, 'a') as f:
            f.write(msg + '\n')
    
    log(f"{'='*70}")
    log(f"TRAINING START")
    log(f"{'='*70}\n")
    
    # Training loop
    for epoch in range(1, config.EPOCHS + 1):
        log(f"\n{'='*70}")
        log(f"Epoch {epoch}/{config.EPOCHS}")
        log(f"{'='*70}")
        
        # Warmup
        warmup = epoch <= config.WARMUP_EPOCHS
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, warmup=warmup)
        log(f"\nTrain Loss: {train_metrics['loss']:.4f}, Macro F1: {train_metrics['macro_f1']:.4f}")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        log(f"Val   Loss: {val_metrics['loss']:.4f}, Macro F1: {val_metrics['macro_f1']:.4f}")
        
        # Log metrics
        for key, value in train_metrics.items():
            if key != 'confusion_matrix':
                history[f'train_{key}'].append(float(value) if not isinstance(value, np.ndarray) else value.tolist())
        
        for key, value in val_metrics.items():
            if key != 'confusion_matrix':
                history[f'val_{key}'].append(float(value) if not isinstance(value, np.ndarray) else value.tolist())
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_metrics[config.CHECKPOINT_METRIC])
        
        # Checkpoint
        current_metric = val_metrics[config.CHECKPOINT_METRIC]
        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0
            
            # Save checkpoint
            checkpoint_path = config.CHECKPOINTS_DIR / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_metric,
                'val_metrics': val_metrics,
                # 'config': {k: v for k, v in vars(config).items() if not k.startswith('__')},
            }, checkpoint_path)
            
            log(f"\n✅ New best {config.CHECKPOINT_METRIC}: {current_metric:.4f} (saved)")
        else:
            patience_counter += 1
            log(f"\n⏳ No improvement ({patience_counter}/{config.EARLY_STOPPING_PATIENCE})")
        
        # Check early stopping
        if patience_counter > config.EARLY_STOPPING_PATIENCE:
            log(f"\n🛑 Early stopping triggered after {epoch} epochs")
            break
    
    log(f"\n{'='*70}")
    log(f"TRAINING COMPLETE")
    log(f"{'='*70}")
    log(f"Best {config.CHECKPOINT_METRIC}: {best_metric:.4f}")
    
    # Load best model
    checkpoint = torch.load(config.CHECKPOINTS_DIR / 'best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final test evaluation
    log(f"\n{'='*70}")
    log("FINAL TEST EVALUATION")
    log(f"{'='*70}\n")
    
    test_metrics = validate(model, test_loader, criterion, device)
    print_metrics(test_metrics, split="Test")
    
    # Save history
    history_path = config.METRICS_DIR / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    log(f"\n📊 Training history saved to {history_path}")
    
    # FINAL EVALUATION
    total_time = time.time() - start_time
    log(f"\n⏱️  Total training time: {total_time / 60:.2f} minutes")
    log(f"📁 Best model saved to {config.CHECKPOINTS_DIR / 'best_model.pt'}")
    log(f"📝 Training log saved to {log_file}")
    
    return model, history, test_metrics


# ============================================
# ENTRY POINT
# ============================================

if __name__ == '__main__':
    # Set reproducibility
    config.set_seed()
    
    # Train
    model, history, test_metrics = train()
    
    print("\n🎉 Training pipeline complete!")
