# """
# Training script for Heterogeneous Temporal GNN (Ransomware Detection).

# Implements FULL POWER training strategy:
# 1. Focal Loss with boosted class weights
# 2. Weighted oversampling for class balance
# 3. Multi-metric evaluation (per-class F1, macro F1, recall, precision)
# 4. Early stopping on macro F1
# 5. Learning rate scheduler (ReduceLROnPlateau)
# 6. Warmup epochs
# 7. Two-stage training (optional)
# 8. Gradient clipping
# 9. Comprehensive logging and checkpointing

# Author: Your Name
# Date: 2026-02-12
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, WeightedRandomSampler
# from torch.optim import AdamW
# from torch.optim.lr_scheduler import ReduceLROnPlateau
# import numpy as np
# from pathlib import Path
# import sys
# import json
# import time
# from datetime import datetime
# from tqdm import tqdm
# from collections import defaultdict
# import warnings
# warnings.filterwarnings('ignore')

# # Add project root
# PROJECT_ROOT = Path(__file__).parent.parent
# sys.path.append(str(PROJECT_ROOT))

# import config
# from models.tgnn_model import create_hetero_tgnn
# from graph_construction.pytorch_dataset import RansomwareGraphDataset, hetero_collate

# # Sklearn metrics
# from sklearn.metrics import (
#     accuracy_score,
#     precision_recall_fscore_support,
#     confusion_matrix,
#     classification_report,
# )


# # ============================================
# # FOCAL LOSS IMPLEMENTATION
# # ============================================

# class FocalLoss(nn.Module):
#     """
#     Focal Loss for addressing class imbalance.
    
#     FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
#     Args:
#         alpha: Class weights [C] (e.g., [0.5, 0.35, 0.15])
#         gamma: Focusing parameter (2.0 recommended)
#         reduction: 'mean' or 'sum'
#     """
    
#     def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
#         super().__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
        
#         if alpha is not None:
#             if isinstance(alpha, list):
#                 self.alpha = torch.tensor(alpha)
#             else:
#                 self.alpha = alpha
    
#     def forward(self, inputs, targets):
#         """
#         Args:
#             inputs: [batch_size, num_classes] logits
#             targets: [batch_size] labels (0, 1, 2)
#         """
#         # Get probabilities
#         probs = F.softmax(inputs, dim=1)
        
#         # Get class probabilities
#         class_mask = F.one_hot(targets, num_classes=inputs.size(1))
#         probs = (probs * class_mask).sum(dim=1)  # [batch_size]
        
#         # Compute focal term: (1 - p_t)^gamma
#         focal_weight = (1 - probs) ** self.gamma
        
#         # Compute cross entropy
#         ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
#         # Apply focal weight
#         focal_loss = focal_weight * ce_loss
        
#         # Apply class weights (alpha)
#         if self.alpha is not None:
#             if self.alpha.device != inputs.device:
#                 self.alpha = self.alpha.to(inputs.device)
#             alpha_t = self.alpha[targets]
#             focal_loss = alpha_t * focal_loss
        
#         # Reduction
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss


# # ============================================
# # METRIC COMPUTATION
# # ============================================

# def compute_metrics(predictions, labels, num_classes=3):
#     """
#     Compute comprehensive metrics.
    
#     Returns dict with:
#     - accuracy
#     - per_class_precision, per_class_recall, per_class_f1
#     - macro_precision, macro_recall, macro_f1
#     - weighted_precision, weighted_recall, weighted_f1
#     - confusion_matrix
#     """
#     # Sklearn metrics
#     acc = accuracy_score(labels, predictions)
    
#     # Per-class and macro/weighted metrics
#     precision, recall, f1, support = precision_recall_fscore_support(
#         labels, predictions, average=None, zero_division=0
#     )
    
#     macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
#         labels, predictions, average='macro', zero_division=0
#     )
    
#     weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
#         labels, predictions, average='weighted', zero_division=0
#     )
    
#     cm = confusion_matrix(labels, predictions)
    
#     metrics = {
#         'accuracy': acc,
#         # Per-class
#         'per_class_precision': precision,
#         'per_class_recall': recall,
#         'per_class_f1': f1,
#         'per_class_support': support,
#         # Macro (equal weight to all classes)
#         'macro_precision': macro_precision,
#         'macro_recall': macro_recall,
#         'macro_f1': macro_f1,
#         # Weighted by class frequency
#         'weighted_precision': weighted_precision,
#         'weighted_recall': weighted_recall,
#         'weighted_f1': weighted_f1,
#         # Confusion matrix
#         'confusion_matrix': cm,
#     }
    
#     return metrics


# def print_metrics(metrics, split="Train", class_names=None):
#     """Pretty print metrics."""
#     if class_names is None:
#         class_names = config.CLASS_NAMES
    
#     print(f"{'='*70}")
#     print(f"{split} Metrics")
#     print(f"{'='*70}")
#     print(f"Accuracy:    {metrics['accuracy']:.4f}")
#     print(f"Macro F1:    {metrics['macro_f1']:.4f}")
#     print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
#     print(f"\n{'Per-Class Metrics'}")
#     print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<12}")
#     print(f"{'-'*70}")
#     for i in range(len(metrics['per_class_precision'])):
#         print(f"{class_names[i]:<20} "
#               f"{metrics['per_class_precision'][i]:<12.4f} "
#               f"{metrics['per_class_recall'][i]:<12.4f} "
#               f"{metrics['per_class_f1'][i]:<12.4f} "
#               f"{int(metrics['per_class_support'][i]):<12}")
    
#     print(f"\n{'Confusion Matrix'}")
#     print(metrics['confusion_matrix'])
#     print(f"{'='*70}")


# # ============================================
# # DATA LOADING
# # ============================================

# def create_dataloaders(use_weighted_sampling=True):
#     """
#     Create train/val/test dataloaders.
    
#     Args:
#         use_weighted_sampling: If True, use WeightedRandomSampler for training
    
#     Returns:
#         train_loader, val_loader, test_loader
#     """
#     print("Loading datasets...")
    
#     # Load datasets
#     train_dataset = RansomwareGraphDataset(
#         data_dir=config.ENRICHED_DIR,
#         split='train'
#     )
    
#     val_dataset = RansomwareGraphDataset(
#         data_dir=config.ENRICHED_DIR,
#         split='val'
#     )
    
#     test_dataset = RansomwareGraphDataset(
#         data_dir=config.ENRICHED_DIR,
#         split='test'
#     )
    
#     print(f"Train: {len(train_dataset)} graphs")
#     print(f"Val:   {len(val_dataset)} graphs")
#     print(f"Test:  {len(test_dataset)} graphs")
    
#     # Count class distribution
#     train_labels = [train_dataset[i][1] for i in range(len(train_dataset))]
#     print(f"\n{'Train class distribution'}")
#     for cls in range(config.NUM_CLASSES):
#         count = train_labels.count(cls)
#         print(f"  {config.CLASS_NAMES[cls]}: {count} ({100 * count / len(train_labels):.2f}%)")
    
#     # Create samplers
#     if use_weighted_sampling and config.USE_WEIGHTED_SAMPLING:
#         print(f"\n⚡ Using weighted oversampling...")
#         sample_weights = []
#         for idx in range(len(train_dataset)):
#             label = train_dataset[idx][1]
#             weight = config.OVERSAMPLE_WEIGHTS.get(label, 1.0)
#             sample_weights.append(weight)
        
#         sampler = WeightedRandomSampler(
#             weights=sample_weights,
#             num_samples=len(train_dataset),
#             replacement=True
#         )
        
#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=config.BATCH_SIZE,
#             sampler=sampler,
#             collate_fn=hetero_collate,
#             num_workers=config.NUM_WORKERS,
#             pin_memory=config.PIN_MEMORY
#         )
#     else:
#         print(f"\n⚡ Using standard sampling (no oversampling)...")
#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=config.BATCH_SIZE,
#             shuffle=True,
#             collate_fn=hetero_collate,
#             num_workers=config.NUM_WORKERS,
#             pin_memory=config.PIN_MEMORY
#         )
    
#     # Val and test (no sampling)
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=config.BATCH_SIZE,
#         shuffle=False,
#         collate_fn=hetero_collate,
#         num_workers=config.NUM_WORKERS,
#         pin_memory=config.PIN_MEMORY
#     )
    
#     test_loader = DataLoader(
#         test_dataset,
#         batch_size=config.BATCH_SIZE,
#         shuffle=False,
#         collate_fn=hetero_collate,
#         num_workers=config.NUM_WORKERS,
#         pin_memory=config.PIN_MEMORY
#     )
    
#     return train_loader, val_loader, test_loader


# # ============================================
# # TRAINING & VALIDATION
# # ============================================

# def train_epoch(model, loader, criterion, optimizer, device, epoch, warmup=False):
#     """Train for one epoch."""
#     model.train()
    
#     total_loss = 0
#     all_preds = []
#     all_labels = []
    
#     pbar = tqdm(loader, desc=f"{'WARMUP ' if warmup else ''}Epoch {epoch}")
    
#     for batch_idx, (batch_data, labels) in enumerate(pbar):
#         # Move to device
#         labels = labels.to(device)
#         batch_data = batch_data.to(device)

#         # ✅ FIXED: Safe timestamp extraction using x_dict keys
#         timestamps_dict = None
#         try:
#             timestamps_dict = {}
#             for node_type in batch_data.x_dict.keys():
#                 if hasattr(batch_data[node_type], 'timestamps'):
#                     timestamps_dict[node_type] = batch_data[node_type].timestamps
            
#             if not timestamps_dict:
#                 timestamps_dict = None
#         except (KeyError, AttributeError):
#             timestamps_dict = None
        
#         # Forward
#         optimizer.zero_grad()
#         logits = model(
#             x_dict=batch_data.x_dict,
#             edge_index_dict=batch_data.edge_index_dict,
#             edge_attr_dict=batch_data.edge_attr_dict,
#             batch_dict=batch_data.batch_dict,
#             timestamps_dict=timestamps_dict
#         )
        
#         # Loss
#         loss = criterion(logits, labels)
        
#         total_loss += loss.item()
#         preds = logits.argmax(dim=1).cpu().numpy()
#         all_preds.extend(preds)
#         all_labels.extend(labels.cpu().numpy())
        
#         # Backward
#         loss.backward()
        
#         # Gradient clipping
#         if config.GRAD_CLIP > 0:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        
#         optimizer.step()
        
#         # Update progress bar
#         pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
#     # Compute metrics
#     avg_loss = total_loss / len(loader)
#     metrics = compute_metrics(all_preds, all_labels)
#     metrics['loss'] = avg_loss
    
#     return metrics

# def validate(model, loader, criterion, device):
#     """Validate."""
#     model.eval()
    
#     total_loss = 0
#     all_preds = []
#     all_labels = []
    
#     with torch.no_grad():
#         for batch_data, labels in tqdm(loader, desc="Validating"):
#             labels = labels.to(device)
#             batch_data = batch_data.to(device)
            
#             # ✅ FIXED: Safe timestamp extraction using x_dict keys
#             timestamps_dict = None
#             try:
#                 timestamps_dict = {}
#                 for node_type in batch_data.x_dict.keys():
#                     if hasattr(batch_data[node_type], 'timestamps'):
#                         timestamps_dict[node_type] = batch_data[node_type].timestamps
                
#                 if not timestamps_dict:
#                     timestamps_dict = None
#             except (KeyError, AttributeError):
#                 timestamps_dict = None
            
#             # Forward
#             logits = model(
#                 x_dict=batch_data.x_dict,
#                 edge_index_dict=batch_data.edge_index_dict,
#                 edge_attr_dict=batch_data.edge_attr_dict,
#                 batch_dict=batch_data.batch_dict,
#                 timestamps_dict=timestamps_dict
#             )
            
#             # Loss
#             loss = criterion(logits, labels)
#             total_loss += loss.item()
#             preds = logits.argmax(dim=1).cpu().numpy()
#             all_preds.extend(preds)
#             all_labels.extend(labels.cpu().numpy())
    
#     # Convert to numpy arrays
#     all_preds = np.array(all_preds)
#     all_labels = np.array(all_labels)
    
#     # 🔍 DEBUG: Print prediction distribution
#     print("\n" + "="*60)
#     print("🔍 VALIDATION DEBUG:")
#     pred_unique, pred_counts = np.unique(all_preds, return_counts=True)
#     label_unique, label_counts = np.unique(all_labels, return_counts=True)
#     print(f"Predicted: {dict(zip(pred_unique, pred_counts))}")
#     print(f"Actual:    {dict(zip(label_unique, label_counts))}")
#     print("="*60 + "\n")
    
#     # Compute metrics
#     avg_loss = total_loss / len(loader)
#     metrics = compute_metrics(all_preds, all_labels)
#     metrics['loss'] = avg_loss
    
#     return metrics





# # ============================================
# # MAIN TRAINING LOOP
# # ============================================

# # def train():
# #     """Main training function."""
    
# #     start_time = time.time()
    
# #     # Device
# #     device = torch.device(config.DEVICE)
# #     print(f"\n{'='*70}")
# #     print(f"Device: {device}")
    
# #     # Create dataloaders
# #     train_loader, val_loader, test_loader = create_dataloaders(
# #         use_weighted_sampling=config.USE_WEIGHTED_SAMPLING
# #     )
    
# #     # Create model
# #     model = create_hetero_tgnn()
# #     model = model.to(device)
    
# #     print(f"\n🚀 Model architecture:")
# #     print(f" API output per layer: {model.api_out_dim_per_layer}")
# #     print(f" Resource output per layer: {model.resource_out_dim_per_layer}")
# #     print(f" LSTM output: {model.lstm_output_dim}")
# #     print(f" Total embedding: {model.total_embedding_dim}")
# #     print(f" Classifier hidden: {model.classifier_hidden_dim}")
    
# #     # Loss function (Focal Loss)
# #     criterion = FocalLoss(
# #         alpha=config.FOCAL_ALPHA,
# #         gamma=config.FOCAL_GAMMA
# #     )
    
# #     # Optimizer
# #     optimizer = AdamW(
# #         model.parameters(),
# #         lr=config.LEARNING_RATE,
# #         weight_decay=config.WEIGHT_DECAY
# #     )
    
# #     # Learning rate scheduler
# #     if config.USE_LR_SCHEDULER:
# #         print(f" Using {config.LR_SCHEDULER_TYPE} scheduler")
# #         if config.LR_SCHEDULER_TYPE == 'ReduceLROnPlateau':
# #             scheduler = ReduceLROnPlateau(
# #                 optimizer,
# #                 mode='max',
# #                 factor=config.LR_SCHEDULER_FACTOR,
# #                 patience=config.LR_SCHEDULER_PATIENCE,
# #                 min_lr=config.LR_MIN,
# #             )
# #         else:
# #             scheduler = None
# #     else:
# #         scheduler = None
    
# #     # Training state
# #     best_metric = 0.0
# #     patience_counter = 0
# #     history = defaultdict(list)
    
# #     # Training log file
# #     log_file = config.LOGS_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
# #     def log(msg):
# #         """Log to console and file."""
# #         print(msg)
# #         with open(log_file, 'a') as f:
# #             f.write(msg + '\n')
    
# #     log(f"{'='*70}")
# #     log(f"TRAINING START")
# #     log(f"{'='*70}\n")
    
# #     # Training loop
# #     for epoch in range(1, config.EPOCHS + 1):
# #         log(f"\n{'='*70}")
# #         log(f"Epoch {epoch}/{config.EPOCHS}")
# #         log(f"{'='*70}")
        
# #         # Warmup
# #         warmup = epoch <= config.WARMUP_EPOCHS
        
# #         # Train
# #         train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, warmup=warmup)
# #         log(f"\nTrain Loss: {train_metrics['loss']:.4f}, Macro F1: {train_metrics['macro_f1']:.4f}")
        
# #         # Validate
# #         val_metrics = validate(model, val_loader, criterion, device)
# #         log(f"Val   Loss: {val_metrics['loss']:.4f}, Macro F1: {val_metrics['macro_f1']:.4f}")
        
# #         # Log metrics
# #         for key, value in train_metrics.items():
# #             if key != 'confusion_matrix':
# #                 history[f'train_{key}'].append(float(value) if not isinstance(value, np.ndarray) else value.tolist())
        
# #         for key, value in val_metrics.items():
# #             if key != 'confusion_matrix':
# #                 history[f'val_{key}'].append(float(value) if not isinstance(value, np.ndarray) else value.tolist())
        
# #         # Learning rate scheduling
# #         if scheduler is not None:
# #             scheduler.step(val_metrics[config.CHECKPOINT_METRIC])
        
# #         # Checkpoint
# #         current_metric = val_metrics[config.CHECKPOINT_METRIC]
# #         if current_metric > best_metric:
# #             best_metric = current_metric
# #             patience_counter = 0
            
# #             # Save checkpoint
# #             checkpoint_path = config.CHECKPOINTS_DIR / 'best_model.pt'
# #             torch.save({
# #                 'epoch': epoch,
# #                 'model_state_dict': model.state_dict(),
# #                 'optimizer_state_dict': optimizer.state_dict(),
# #                 'best_metric': best_metric,
# #                 'val_metrics': val_metrics,
# #                 # 'config': {k: v for k, v in vars(config).items() if not k.startswith('__')},
# #             }, checkpoint_path)
            
# #             log(f"\n✅ New best {config.CHECKPOINT_METRIC}: {current_metric:.4f} (saved)")
# #         else:
# #             patience_counter += 1
# #             log(f"\n⏳ No improvement ({patience_counter}/{config.EARLY_STOPPING_PATIENCE})")
        
# #         # Check early stopping
# #         if patience_counter > config.EARLY_STOPPING_PATIENCE:
# #             log(f"\n🛑 Early stopping triggered after {epoch} epochs")
# #             break
    
# #     log(f"\n{'='*70}")
# #     log(f"TRAINING COMPLETE")
# #     log(f"{'='*70}")
# #     log(f"Best {config.CHECKPOINT_METRIC}: {best_metric:.4f}")
    
# #     # Load best model
# #     checkpoint = torch.load(config.CHECKPOINTS_DIR / 'best_model.pt')
# #     model.load_state_dict(checkpoint['model_state_dict'])
    
# #     # Final test evaluation
# #     log(f"\n{'='*70}")
# #     log("FINAL TEST EVALUATION")
# #     log(f"{'='*70}\n")
    
# #     test_metrics = validate(model, test_loader, criterion, device)
# #     print_metrics(test_metrics, split="Test")
    
# #     # Save history
# #     history_path = config.METRICS_DIR / 'training_history.json'
# #     with open(history_path, 'w') as f:
# #         json.dump(history, f, indent=2)
    
# #     log(f"\n📊 Training history saved to {history_path}")
    
# #     # FINAL EVALUATION
# #     total_time = time.time() - start_time
# #     log(f"\n⏱️  Total training time: {total_time / 60:.2f} minutes")
# #     log(f"📁 Best model saved to {config.CHECKPOINTS_DIR / 'best_model.pt'}")
# #     log(f"📝 Training log saved to {log_file}")
    
# #     return model, history, test_metrics


# # # ============================================
# # # ENTRY POINT
# # # ============================================

# # if __name__ == '__main__':
# #     # Set reproducibility
# #     config.set_seed()
    
# #     # Train
# #     model, history, test_metrics = train()
    
# #     print("\n🎉 Training pipeline complete!")


# def train():
#     """Main training function."""
    
#     start_time = time.time()
    
#     # Device
#     device = torch.device(config.DEVICE)
#     print(f"\n{'='*70}")
#     print(f"Device: {device}")
    
#     # Create dataloaders
#     train_loader, val_loader, test_loader = create_dataloaders(
#         use_weighted_sampling=config.USE_WEIGHTED_SAMPLING
#     )
    
#     # Create model
#     model = create_hetero_tgnn()
#     model = model.to(device)
    
#     print(f"\n🚀 Model architecture:")
#     print(f" API output per layer: {model.api_out_dim_per_layer}")
#     print(f" Resource output per layer: {model.resource_out_dim_per_layer}")
#     print(f" LSTM output: {model.lstm_output_dim}")
#     print(f" Total embedding: {model.total_embedding_dim}")
#     print(f" Classifier hidden: {model.classifier_hidden_dim}")
    
#     # Loss function (Focal Loss)
#     criterion = FocalLoss(
#         alpha=config.FOCAL_ALPHA,
#         gamma=config.FOCAL_GAMMA
#     )
    
#     # Optimizer
#     optimizer = AdamW(
#         model.parameters(),
#         lr=config.LEARNING_RATE,
#         weight_decay=config.WEIGHT_DECAY
#     )
    
#     # Learning rate scheduler
#     if config.USE_LR_SCHEDULER:
#         print(f" Using {config.LR_SCHEDULER_TYPE} scheduler")
#         if config.LR_SCHEDULER_TYPE == 'ReduceLROnPlateau':
#             scheduler = ReduceLROnPlateau(
#                 optimizer,
#                 mode='max',
#                 factor=config.LR_SCHEDULER_FACTOR,
#                 patience=config.LR_SCHEDULER_PATIENCE,
#                 min_lr=config.LR_MIN,
#             )
#         else:
#             scheduler = None
#     else:
#         scheduler = None
    
#     # Training state
#     best_metric = 0.0
#     patience_counter = 0
#     history = defaultdict(list)

#     # ✅ Resume from existing checkpoint ONCE before epoch 1
#     checkpoint_path = config.CHECKPOINTS_DIR / 'best_model.pt'
#     start_epoch = 1

#     if checkpoint_path.exists():
#         checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
#         model.load_state_dict(checkpoint['model_state_dict'])

#         # optional but recommended if you want true resume (keeps AdamW momentum, etc.)
#         if 'optimizer_state_dict' in checkpoint:
#             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

#         best_metric = float(checkpoint.get('best_metric', 0.0))
#         start_epoch = int(checkpoint.get('epoch', 0)) + 1

#         print(f"🔁 Loaded checkpoint: {checkpoint_path}")
#         print(f"✅ Resuming from epoch {start_epoch} | best {config.CHECKPOINT_METRIC}={best_metric:.4f}")
#     else:
#         print("ℹ️ No checkpoint found. Training from scratch.")
    
#     # Training log file
#     log_file = config.LOGS_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
#     def log(msg):
#         """Log to console and file."""
#         print(msg)
#         with open(log_file, 'a') as f:
#             f.write(msg + '\n')
    
#     log(f"{'='*70}")
#     log(f"TRAINING START")
#     log(f"{'='*70}\n")
    
#     # Training loop
#     for epoch in range(start_epoch, config.EPOCHS + 1):
#         log(f"\n{'='*70}")
#         log(f"Epoch {epoch}/{config.EPOCHS}")
#         log(f"{'='*70}")
        
#         # Warmup
#         warmup = epoch <= config.WARMUP_EPOCHS
        
#         # Train
#         train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch, warmup=warmup)
#         log(f"\nTrain Loss: {train_metrics['loss']:.4f}, Macro F1: {train_metrics['macro_f1']:.4f}")
        
#         # Validate
#         val_metrics = validate(model, val_loader, criterion, device)
#         log(f"Val   Loss: {val_metrics['loss']:.4f}, Macro F1: {val_metrics['macro_f1']:.4f}")
        
#         # Log metrics
#         for key, value in train_metrics.items():
#             if key != 'confusion_matrix':
#                 history[f'train_{key}'].append(float(value) if not isinstance(value, np.ndarray) else value.tolist())
        
#         for key, value in val_metrics.items():
#             if key != 'confusion_matrix':
#                 history[f'val_{key}'].append(float(value) if not isinstance(value, np.ndarray) else value.tolist())
        
#         # Learning rate scheduling
#         if scheduler is not None:
#             scheduler.step(val_metrics[config.CHECKPOINT_METRIC])
        
#         # Checkpoint
#         current_metric = val_metrics[config.CHECKPOINT_METRIC]
#         if current_metric > best_metric:
#             best_metric = current_metric
#             patience_counter = 0
            
#             # Save checkpoint
#             checkpoint_path = config.CHECKPOINTS_DIR / 'best_model.pt'
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'best_metric': best_metric,
#                 'val_metrics': val_metrics,
#                 # 'config': {k: v for k, v in vars(config).items() if not k.startswith('__')},
#             }, checkpoint_path)
            
#             log(f"\n✅ New best {config.CHECKPOINT_METRIC}: {current_metric:.4f} (saved)")
#         else:
#             patience_counter += 1
#             log(f"\n⏳ No improvement ({patience_counter}/{config.EARLY_STOPPING_PATIENCE})")
        
#         # Check early stopping
#         if patience_counter > config.EARLY_STOPPING_PATIENCE:
#             log(f"\n🛑 Early stopping triggered after {epoch} epochs")
#             break
    
#     log(f"\n{'='*70}")
#     log(f"TRAINING COMPLETE")
#     log(f"{'='*70}")
#     log(f"Best {config.CHECKPOINT_METRIC}: {best_metric:.4f}")
    
#     # Load best model
#     checkpoint = torch.load(config.CHECKPOINTS_DIR / 'best_model.pt')
#     model.load_state_dict(checkpoint['model_state_dict'])
    
#     # Final test evaluation
#     log(f"\n{'='*70}")
#     log("FINAL TEST EVALUATION")
#     log(f"{'='*70}\n")
    
#     test_metrics = validate(model, test_loader, criterion, device)
#     print_metrics(test_metrics, split="Test")
    
#     # Save history
#     history_path = config.METRICS_DIR / 'training_history.json'
#     with open(history_path, 'w') as f:
#         json.dump(history, f, indent=2)
    
#     log(f"\n📊 Training history saved to {history_path}")
    
#     # FINAL EVALUATION
#     total_time = time.time() - start_time
#     log(f"\n⏱️  Total training time: {total_time / 60:.2f} minutes")
#     log(f"📁 Best model saved to {config.CHECKPOINTS_DIR / 'best_model.pt'}")
#     log(f"📝 Training log saved to {log_file}")
    
#     return model, history, test_metrics


# # # ============================================
# # # ENTRY POINT
# # # ============================================

# if __name__ == '__main__':
#     # Set reproducibility
#     config.set_seed()
    
#     # Train
#     model, history, test_metrics = train()
    
#    print("\n🎉 Training pipeline complete!")

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

def train_epoch(model, loader, criterion, optimizer, vocab, device, epoch, warmup=False):
    """Train for one epoch — with memory write after every optimizer step."""
    model.train()

    total_loss    = 0.0
    total_mem_reg = 0.0
    all_preds     = []
    all_labels    = []

    pbar = tqdm(loader, desc=f"{'WARMUP ' if warmup else ''}Epoch {epoch}")

    for batch_data, labels in pbar:
        labels     = labels.to(device)
        batch_data = batch_data.to(device)

        timestamps_dict = _extract_timestamps(batch_data)

        # ── MEMORY (1): build api_indices from API name strings ───────────────
        api_indices = _extract_api_indices(batch_data, vocab, device)

        optimizer.zero_grad()

        # ── forward ───────────────────────────────────────────────────────────
        logits = model(
            x_dict          = batch_data.x_dict,
            edge_index_dict = batch_data.edge_index_dict,
            edge_attr_dict  = batch_data.edge_attr_dict,
            batch_dict      = batch_data.batch_dict,
            timestamps_dict = timestamps_dict,
            api_indices     = api_indices,           # ← MEMORY (1)
        )

        # ── MEMORY (2): add memory regularisation to task loss ────────────────
        task_loss = criterion(logits, labels)
        mem_reg   = model.memory_loss(lambda_reg=config.MEMORY_LAMBDA)   # ← MEMORY (2)
        loss      = task_loss + mem_reg                  # ← MEMORY (2)

        total_loss    += task_loss.item()
        total_mem_reg += mem_reg.item()

        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        # ── backward + clip ───────────────────────────────────────────────────
        loss.backward()
        if config.GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()

        # ── MEMORY (3): write new knowledge into memory bank ─────────────────
        with torch.no_grad(): 
            model.update_memory(                             # ← MEMORY (3)
                api_indices   = api_indices,
                # node_features = batch_data['api'].x,        # raw 775-dim, before enrichment
                node_features=batch_data['api'].x.detach(),
                momentum      = config.MEMORY_MOMENTUM,
            )
        # raw_mem_norm = model.memory.norm().item() 
        # raw_mem_norm = model.memory.memory_bank.norm().item()
        raw_mem_norm = model.memory.memory_bank.weight.norm().item()
        # print(dict(model.memory.named_parameters()))
        pbar.set_postfix(
            loss=f"{task_loss.item():.4f}",
            mem_reg=f"{mem_reg.item():.1e}",
            mem_norm=f"{raw_mem_norm:.1e}",
            lr=f"{optimizer.param_groups[0]['lr']:.1e}"
        )
    

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
            vocab, device, epoch, warmup=warmup,       # ← MEMORY: vocab passed
        )
        log(f"\nTrain  loss={train_metrics['loss']:.4f}  "
            f"macro_f1={train_metrics['macro_f1']:.4f}  "
            f"mem_reg={train_metrics['mem_reg']:.6f}")  # ← MEMORY: log reg term

        # validate (no memory write)
        val_metrics = validate(model, val_loader, criterion, vocab, device)
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
        if current > best_metric:
            best_metric      = current
            patience_counter = 0
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric':          best_metric,
                'val_metrics':          val_metrics,
            }, checkpoint_path)
            log(f"\n✅ New best {config.CHECKPOINT_METRIC}: {current:.4f} (saved)")
        else:
            patience_counter += 1
            log(f"\n⏳ No improvement ({patience_counter}/{config.EARLY_STOPPING_PATIENCE})")

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
