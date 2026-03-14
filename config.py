
# set_seed()
"""
Configuration file for Ransomware TGNN project.
"""

import torch
import random
import numpy as np
from pathlib import Path

# ============================================
# REPRODUCIBILITY
# ============================================
RANDOM_SEED = 42

def set_seed(seed=RANDOM_SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Set seed on import
set_seed()


# ============================================
# PATHS
# ============================================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
CACHE_DIR = PROJECT_ROOT / 'cache'
RESULTS_DIR = PROJECT_ROOT / 'results'

# Data subdirectories
SAMPLE_DIR = DATA_DIR / 'sample'
SPLITS_DIR = DATA_DIR / 'splits'
METADATA_PATH = DATA_DIR / 'extracted_metadata.csv'
LOOKUP_PATH = DATA_DIR / 'lookup.json'

# Cache subdirectories
GRAPHS_DIR = CACHE_DIR / 'graphs'
ENRICHED_DIR = CACHE_DIR / 'enriched'

# Results subdirectories
CHECKPOINTS_DIR = RESULTS_DIR / 'checkpoints'
LOGS_DIR = RESULTS_DIR / 'logs'
METRICS_DIR = RESULTS_DIR / 'metrics'

# Create directories
for dir_path in [CACHE_DIR, RESULTS_DIR, CHECKPOINTS_DIR, LOGS_DIR, METRICS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# ============================================
# DATA PROCESSING
# ============================================
MAX_API_CALLS = 10000  # Truncate long sequences
MIN_API_CALLS = 10     # Filter out too-short traces
CHUNK_SIZE = 100       # Graphs per chunk file


# ============================================
# DATASET SPLITS
# ============================================
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================
# GRAPH CONSTRUCTION
# ============================================
# Node types
NODE_TYPES = ['api', 'file', 'registry', 'process', 'network']

# Edge types (source, relation, target)
EDGE_TYPES = [
    ('api', 'temporal', 'api'),           # Temporal edges between API calls
    ('api', 'operates_on', 'file'),       # API operates on file
    ('api', 'modifies', 'registry'),      # API modifies registry
    ('process', 'executes', 'api'),       # Process executes API
    ('api', 'connects_to', 'network'),    # API connects to network
]


# ============================================
# FEATURE DIMENSIONS
# ============================================
# Node features
API_NODE_DIM = 775          # 768 BERT + 7 metadata
RESOURCE_NODE_DIM = 6       # File/Registry/Process/Network features

# Edge features
TEMPORAL_EDGE_DIM = 14      # Temporal edge features


# ============================================
# MODEL ARCHITECTURE
# ============================================
# Spatial aggregation
SPATIAL_LAYER = 'gat'       # Options: 'gat', 'transformer'
HIDDEN_DIM = 128            # Hidden dimension for GNN layers
RESOURCE_HIDDEN_DIM = 64    # Resource node hidden dimension
NUM_GNN_LAYERS = 2          # Number of spatial aggregation layers
NUM_ATTENTION_HEADS = 6     # Multi-head attention (spatial)
DROPOUT = 0.4               # Dropout rate

# Temporal modeling
LSTM_HIDDEN_DIM = 126       # LSTM hidden dimension
LSTM_NUM_LAYERS = 2         # Number of LSTM layers
BIDIRECTIONAL = True        # Use bidirectional LSTM
LSTM_DROPOUT = 0.2          # LSTM dropout

# Temporal attention pooling
TEMPORAL_HEADS = 6          # Multi-head attention (temporal)

# Output
NUM_CLASSES = 3             # Ransomware, Benign, Other Malware


# ============================================
# TRAINING HYPERPARAMETERS
# ============================================
# Optimization
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
BATCH_SIZE = 32
EPOCHS = 125
MAX_EPOCHS = 135

# Early stopping
EARLY_STOPPING_PATIENCE = 20
CHECKPOINT_METRIC = 'macro_f1'  # Options: 'macro_f1', 'val_loss', 'accuracy'

# Gradient clipping
GRAD_CLIP = 1.0

# Warmup
USE_WARMUP = True
WARMUP_EPOCHS = 5
WARMUP_LR = 1e-5

# Device
DEVICE = (
    'cuda' if torch.cuda.is_available() 
    # else 'mps' if torch.backends.mps.is_available() 
    else 'cpu'
)
# DataLoader
NUM_WORKERS = 0 # -- was 4
PIN_MEMORY = True if DEVICE == 'cuda' else False

RESUME_FROM_CHECKPOINT = True  
# ═══════════════════════════════════════════════════════════
# DATALOADER (macOS FIX)
# ═══════════════════════════════════════════════════════════
# NUM_WORKERS = 0  # MUST be 0 on macOS!

# ============================================
# CLASS IMBALANCE HANDLING
# ============================================
# Dataset distribution: Ransomware=173, Benign=68, Other=4356

# Focal Loss
USE_FOCAL_LOSS = True
# FOCAL_GAMMA = 2.0
FOCAL_GAMMA = 2
# FOCAL_ALPHA = [0.40, 0.50, 0.10]  # [Ransomware, Benign, Other Malware]
# FOCAL_ALPHA = [0.38, 0.50, 0.12]
FOCAL_ALPHA = [0.25, 0.35, 0.40]
# Weighted Sampling (Oversampling)
USE_WEIGHTED_SAMPLING = True
# OVERSAMPLE_WEIGHTS = {
#     0: 6.0,   # Ransomware: 35x oversampling
#     1: 14.0,   # Benign: 63x oversampling
#     2: 0.8    # Other Malware: 1.2x oversampling
# }
OVERSAMPLE_WEIGHTS = {
    0: 6.0,
    1: 6.0,
    2: 0.6
}
# Class weights for loss function (alternative to Focal Loss)
CLASS_WEIGHTS = {
    0: 1.0 / (173 ** 0.5),   # Ransomware: ~0.076
    1: 1.0 / (68 ** 0.5),    # Benign: ~0.121
    2: 1.0 / (4356 ** 0.5),  # Other: ~0.015
}
# Normalize
total = sum(CLASS_WEIGHTS.values())
CLASS_WEIGHTS = {k: v/total for k, v in CLASS_WEIGHTS.items()}


# ============================================
# LEARNING RATE SCHEDULER
# ============================================
USE_LR_SCHEDULER = True
LR_SCHEDULER_TYPE = 'ReduceLROnPlateau'
LR_SCHEDULER_PATIENCE = 5      # Reduce LR if no improvement for 5 epochs
LR_SCHEDULER_FACTOR = 0.5      # LR = LR * 0.5
LR_MIN = 1e-6                  # Minimum learning rate


# ============================================
# ADVANCED TRAINING (OPTIONAL)
# ============================================
# Two-Stage Training # currently not used
USE_TWO_STAGE_TRAINING = False
STAGE1_EPOCHS = 30             # Balanced oversampling
STAGE2_EPOCHS = 50             # Original distribution
STAGE2_LR = 1e-5


# ============================================
# EVALUATION
# ============================================
# Metrics to compute
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'confusion_matrix', 'roc_auc']

# Class names
CLASS_NAMES = {
    0: 'Ransomware',
    1: 'Benign',
    2: 'Other Malwares'
}

# Minimum acceptable metrics (for validation)
MIN_RANSOMWARE_RECALL = 0.95   # Never miss ransomware
MIN_BENIGN_RECALL = 0.90       # High recall for benign
MIN_RANSOMWARE_PRECISION = 0.70  # Alert if too many false positives

# Confidence thresholds (post-training adjustment)
PREDICTION_THRESHOLDS = {
    0: 0.3,  # Ransomware: lower threshold (more sensitive)
    1: 0.4,  # Benign
    2: 0.5   # Other Malware
}


# ============================================
# LOGGING & CHECKPOINTING
# ============================================
LOG_INTERVAL = 10           # Log every N batches
SAVE_BEST_ONLY = True       # Only save best validation model


# config.py — confirm / add these lines
MAX_VOCAB_SIZE  = 5000      # already there 
VOCAB_SAVE_PATH = PROJECT_ROOT / "results" / "api_vocabulary.json"  # ADD if missing
USE_MEMORY    = True
MEMORY_DIM    = 128
MEMORY_MOMENTUM = 0.9
MEMORY_LAMBDA   = 1e-4

# USE_EMA    = True
# EMA_DECAY  = 0.999
OOM_ABORT_THRESHOLD = 12