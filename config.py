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
# Metadata CSV



# ─────────────────────────────────────────────
# Temporal backbone configuration
# ─────────────────────────────────────────────

# Node type that acts as the temporal sequence backbone
# CAPE graphs use API calls as the temporal spine.
PRIMARY_TEMPORAL_NODE_TYPE = "api"
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"

# Data subdirectories
SAMPLE_DIR = DATA_DIR / "sample"
SPLITS_DIR = DATA_DIR / "splits"
# METADATA_PATH = DATA_DIR / "extracted_metadata.csv"
LOOKUP_PATH = DATA_DIR / "lookup.json"
ROOT_METADATA_PATH = PROJECT_ROOT / "extracted_metadata_appended.csv"
DATA_METADATA_PATH = DATA_DIR / "extracted_metadata_appended.csv"

METADATA_PATH = ROOT_METADATA_PATH if ROOT_METADATA_PATH.exists() else DATA_METADATA_PATH
# Cache subdirectories
GRAPHS_DIR = CACHE_DIR / "graphs"
ENRICHED_DIR = CACHE_DIR / "enriched"  # legacy/default enriched path

# Results subdirectories
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
LOGS_DIR = RESULTS_DIR / "logs"
METRICS_DIR = RESULTS_DIR / "metrics"

# ============================================
# GRAPH SOURCES / SCHEMA REGISTRY
# ============================================
# Keep one registry so training/inference can switch graph families
# without rewriting path logic elsewhere.
GRAPH_SOURCES = {
    "cape_enriched": CACHE_DIR / "enriched",
    # Add in Phase 2 when HA-enriched graphs are ready:
    # "ha_enriched": CACHE_DIR / "ha_enriched",
}

# Active graph family for current run
ACTIVE_GRAPH_SOURCE = "cape_enriched" # we can change this to "ha_enriched" later when we have those graphs ready
ACTIVE_GRAPH_DIR = GRAPH_SOURCES[ACTIVE_GRAPH_SOURCE]

# Schema output paths (one schema per graph source)
SCHEMA_DIR = RESULTS_DIR / "schema"
ACTIVE_SCHEMA_DIR = SCHEMA_DIR / ACTIVE_GRAPH_SOURCE
ACTIVE_SCHEMA_PATH = ACTIVE_SCHEMA_DIR / "schema.json"
ACTIVE_SCHEMA_SUMMARY_PATH = ACTIVE_SCHEMA_DIR / "schema_summary.txt"

# Create directories
for dir_path in [
    CACHE_DIR,
    RESULTS_DIR,
    CHECKPOINTS_DIR,
    LOGS_DIR,
    METRICS_DIR,
    SCHEMA_DIR,
    ACTIVE_SCHEMA_DIR,
]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================
# SCHEMA AUTOMATION
# ============================================
AUTO_REFRESH_SCHEMA = True
FAIL_ON_SCHEMA_MISMATCH = True
SCHEMA_STRICT_VALIDATION = True
# Number of successfully processed graphs to generate.
# Set to None to process everything.
MAX_SUCCESSFUL_GRAPHS = 200
# ============================================
# SCHEMA NORMALIZATION POLICY
# ============================================
# Mixed feature dims can exist across cached graphs for the same node type.
# The loader/collate path should normalize against the active schema.
NORMALIZE_NODE_FEATURE_DIMS = True
NORMALIZATION_MODE = "pad_to_schema"   # options: "pad_to_schema", "strict"
ALLOW_FEATURE_TRUNCATION = True       # safer default; avoid silent info loss

# Stable fallback ordering for current graph families.
# schema_utils.py can override with schema-derived ordering later.
DEFAULT_NODE_TYPE_ORDER = ["api", "file", "registry", "process", "network"]

DEFAULT_EDGE_TYPE_ORDER = [
    ("api", "temporal", "api"),
    ("api", "operates_on", "file"),
    ("api", "modifies", "registry"),
    ("process", "executes", "api"),
    ("api", "connects_to", "network"),
]

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
NODE_TYPES = ["api", "file", "registry", "process", "network"]

# Edge types (source, relation, target)
EDGE_TYPES = [
    ("api", "temporal", "api"),           # Temporal edges between API calls
    ("api", "operates_on", "file"),       # API operates on file
    ("api", "modifies", "registry"),      # API modifies registry
    ("process", "executes", "api"),       # Process executes API
    ("api", "connects_to", "network"),    # API connects to network
]

# ============================================
# FEATURE DIMENSIONS
# ============================================
# Current enriched CAPE expectation.
# Phase 1 model upgrade will move toward schema-driven handling,
# but these remain useful defaults/reference checks.
API_NODE_DIM = 775          # 768 BERT + 7 metadata
RESOURCE_NODE_DIM = 6       # Legacy/default resource dim reference
TEMPORAL_EDGE_DIM = 14      # Temporal edge features

# ============================================
# MODEL ARCHITECTURE
# ============================================
# Spatial aggregation
SPATIAL_LAYER = "gat"       # Options: 'gat', 'transformer'
HIDDEN_DIM = 128            # Hidden dimension for GNN layers
RESOURCE_HIDDEN_DIM = 64    # Resource node hidden dimension
NUM_GNN_LAYERS = 2          # Number of spatial aggregation layers
NUM_ATTENTION_HEADS = 6     # Multi-head attention (spatial)
DROPOUT = 0.3               # Dropout rate

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
BATCH_SIZE = 16
EPOCHS = 125
MAX_EPOCHS = 135

# Early stopping
EARLY_STOPPING_PATIENCE = 20    
CHECKPOINT_METRIC = "macro_f1"  # Options: 'macro_f1', 'val_loss', 'accuracy'

# Gradient clipping
GRAD_CLIP = 1.0

# Warmup
USE_WARMUP = True
WARMUP_EPOCHS = 5
WARMUP_LR = 1e-5

# Device
DEVICE = (
    "cuda" if torch.cuda.is_available()
    # else 'mps' if torch.backends.mps.is_available()
    else "cpu"
)

# DataLoader
NUM_WORKERS = 4 if DEVICE == "cuda" else 0  # -- was 4
PREFETCH_FACTOR = 2
PIN_MEMORY = True if DEVICE == "cuda" else False

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
FOCAL_GAMMA = 2
# FOCAL_ALPHA = [0.30, 0.40, 0.30]
# FOCAL_ALPHA = [0.35, 0.35, 0.30] # 
FOCAL_ALPHA = [0.39, 0.39, 0.40] # slight tweak to give more weight to ransomware and less to benign, while keeping other malware dominant
# Weighted Sampling (Oversampling)
USE_WEIGHTED_SAMPLING = True
OVERSAMPLE_WEIGHTS = {
    0: 15, # RANSOMWARE: give it much higher weight to ensure it's well-represented in training batches
    1: 5.0, # BENIGN: moderate weight to improve representation without overwhelming the model
    2: 1.3, # OTHER MALWARE: some oversampling to help model learn better, but not as much as ransomware
}

# Class weights for loss function (alternative to Focal Loss)
CLASS_WEIGHTS = {
    0: 1.0 / (173 ** 0.5),   # Ransomware: ~0.076
    1: 1.0 / (68 ** 0.5),    # Benign: ~0.121
    2: 1.0 / (4356 ** 0.5),  # Other: ~0.015
}

# Normalize
total = sum(CLASS_WEIGHTS.values())
CLASS_WEIGHTS = {k: v / total for k, v in CLASS_WEIGHTS.items()}

# ============================================
# LEARNING RATE SCHEDULER
# ============================================
USE_LR_SCHEDULER = True
LR_SCHEDULER_TYPE = "ReduceLROnPlateau"
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
METRICS = ["accuracy", "precision", "recall", "f1", "confusion_matrix", "roc_auc"]

# Class names
CLASS_NAMES = {
    0: "Ransomware",
    1: "Benign",
    2: "Other Malwares",
}

# Minimum acceptable metrics (for validation)
MIN_RANSOMWARE_RECALL = 0.95      # Never miss ransomware
MIN_BENIGN_RECALL = 0.90          # High recall for benign
MIN_RANSOMWARE_PRECISION = 0.70   # Alert if too many false positives

# Confidence thresholds (post-training adjustment)
PREDICTION_THRESHOLDS = {
    0: 0.3,  # Ransomware: lower threshold (more sensitive)
    1: 0.4,  # Benign
    2: 0.5,  # Other Malware
}

# ============================================
# LOGGING & CHECKPOINTING
# ============================================
LOG_INTERVAL = 10           # Log every N batches
SAVE_BEST_ONLY = True       # Only save best validation model

# ============================================
# MEMORY / VOCAB
# ============================================
MAX_VOCAB_SIZE = 5000
VOCAB_SAVE_PATH = PROJECT_ROOT / "results" / "api_vocabulary.json"

USE_MEMORY = False
MEMORY_DIM = 128
MEMORY_MOMENTUM = 0.85
MEMORY_LAMBDA = 1e-4


# USE_EMA = True
# EMA_DECAY = 0.999
OOM_ABORT_THRESHOLD = 12

# ============================================
# MEMORY V2
# ============================================
MEMORY_WARMUP_EPOCHS = 4

USE_MEMORY_GATED_READ = True
USE_MEMORY_CONTEXT_QUERY = True

MEMORY_GATE_HIDDEN_DIM = 128
MEMORY_READ_TEMPERATURE = 1.0
MEMORY_MIN_GATE = 0.05
MEMORY_MAX_GATE = 0.95

USE_TEMPORAL_CONTEXT_IN_MEMORY = True
USE_NEIGHBOR_CONTEXT_IN_MEMORY = True
USE_LIGHT_GRAPH_CONTEXT_IN_MEMORY = False

MEMORY_QUERY_HIDDEN_DIM = 256
MEMORY_QUERY_OUTPUT_DIM = 128

USE_SELECTIVE_MEMORY_WRITE = True
USE_CONFIDENCE_AWARE_WRITE = True
USE_FREQUENCY_AWARE_WRITE = True
USE_NOVELTY_AWARE_WRITE = True

MEMORY_WRITE_MIN_CONFIDENCE = 0.60
MEMORY_WRITE_BASE_STRENGTH = 1.0
MEMORY_WRITE_MIN_STRENGTH = 0.05
MEMORY_WRITE_MAX_STRENGTH = 1.0

MEMORY_FREQ_ALPHA = 0.5
MEMORY_FREQ_EPS = 1e-6

MEMORY_NOVELTY_TEMPERATURE = 1.0
MEMORY_NOVELTY_EPS = 1e-6

MEMORY_COMMON_TOKEN_WRITE_CAP = 0.25




# ============================================
# API RUN COMPRESSION
# ============================================
# If False:
#   keep graph structure exactly as-is
# If True:
#   compress only strictly redundant consecutive API runs
ENABLE_API_RUN_COMPRESSION = True
API_RUN_COMPRESSION_REQUIRE_API_NAMES = True
API_RUN_COMPRESSION_METADATA_START_DIM = 768
API_RUN_COMPRESSION_METADATA_ATOL = 1e-6
API_RUN_COMPRESSION_USE_TIME_GAP = True
API_RUN_COMPRESSION_MAX_TIME_GAP = 1.0
API_RUN_COMPRESSION_STORE_AUX = True

# ============================================
# BEHAVIOR-AWARE API CAPPING
# ============================================
ENABLE_BEHAVIOR_AWARE_API_CAP = True

# Only cap graphs whose API sequence is larger than this
BEHAVIOR_AWARE_API_CAP_THRESHOLD = 100000

# Cap those graphs down to this many API nodes
BEHAVIOR_AWARE_API_CAP_MAX_NODES = 30000

# Protection knobs
BEHAVIOR_AWARE_API_CAP_PROTECT_EDGE_ENDPOINTS = True
BEHAVIOR_AWARE_API_CAP_PROTECT_TIME_SPIKES = True
BEHAVIOR_AWARE_API_CAP_TIME_SPIKE_QUANTILE = 0.95
BEHAVIOR_AWARE_API_CAP_REQUIRE_API_NAMES = False
TARGETED_API_CAP_MAX_NODES = 100000



# ============================================
# BUCKETED SIZE-AWARE BATCHING
# ============================================

# Policy is based on API node count after any compression / behavior-aware cap.
# Format: (max_api_nodes_in_bucket, batch_size_for_that_bucket)
# Buckets are evaluated in order.
ENABLE_BUCKETED_BATCH_SAMPLER = True
BUCKETED_BATCH_RULES = [
    (10_000, 64),
    (25_000, 16),
    (50_000, 8),
    (100_000, 4),
    (10_000_000, 1),
]
BUCKETED_BATCH_SHUFFLE = True
BUCKETED_BATCH_USE_WEIGHTS = True
BUCKETED_BATCH_VERBOSE = True



# ============================================
# CUSTOM ZERO-DAY / MIXED EVALUATION
# ============================================
USE_CUSTOM_ZERO_DAY_EVAL = True

CUSTOM_EVAL_SOURCE_SPLITS = ["val", "test"]

CUSTOM_EVAL_NEW_MALWARE_DIR = CACHE_DIR / "zero_day_unseen" / "all"

CUSTOM_EVAL_TARGET_BENIGN = 200
CUSTOM_EVAL_TARGET_OLD_MALWARE = 55
CUSTOM_EVAL_INCLUDE_ALL_RANSOMWARE = True
CUSTOM_EVAL_INCLUDE_ALL_NEW_MALWARE = True

# 0 = Ransomware, 1 = Benign, 2 = Other Malwares
CUSTOM_EVAL_NEW_MALWARE_LABEL = 2

CUSTOM_EVAL_DIR = RESULTS_DIR / "custom_eval"
CUSTOM_EVAL_DIR.mkdir(parents=True, exist_ok=True)

CUSTOM_EVAL_MANIFEST_PATH = CUSTOM_EVAL_DIR / "zero_day_manifest.csv"
CUSTOM_EVAL_RESULTS_PATH = CUSTOM_EVAL_DIR / "zero_day_results.json"
CUSTOM_EVAL_PREDICTIONS_PATH = CUSTOM_EVAL_DIR / "zero_day_predictions.csv"

CUSTOM_EVAL_CHECKPOINT_PATH = CHECKPOINTS_DIR / "best_model.pt"

# Separate loader controls for custom evaluation - baseline config
CUSTOM_EVAL_BATCH_SIZE = 1
CUSTOM_EVAL_NUM_WORKERS = 0
CUSTOM_EVAL_PIN_MEMORY = True if DEVICE == "cuda" else False
CUSTOM_EVAL_SHUFFLE = False
CUSTOM_EVAL_VALIDATE_BATCH = False
CUSTOM_EVAL_VALIDATE_GRAPHS = False

CUSTOM_EVAL_REBUILD_MANIFEST = True
CUSTOM_EVAL_SHUFFLE_BEFORE_SAMPLE = True

CUSTOM_EVAL_FAIL_IF_NOT_ENOUGH_BENIGN = True
CUSTOM_EVAL_FAIL_IF_NOT_ENOUGH_OLD_MALWARE = True
CUSTOM_EVAL_FAIL_IF_NEW_MALWARE_DIR_MISSING = True

CUSTOM_EVAL_ENFORCE_NEW_UNSEEN_DISJOINT = True



# zero-day evaluation configuration
CUSTOM_EVAL_SOURCE_SPLITS = ["val", "test"]
CUSTOM_EVAL_NEW_MALWARE_DIR = CACHE_DIR / "zero_day_unseen" / "all"
CUSTOM_EVAL_TARGET_BENIGN = 200
CUSTOM_EVAL_TARGET_OLD_MALWARE = 55
CUSTOM_EVAL_INCLUDE_ALL_RANSOMWARE = True
CUSTOM_EVAL_INCLUDE_ALL_NEW_MALWARE = True
CUSTOM_EVAL_NEW_MALWARE_LABEL = 2
CUSTOM_EVAL_DIR = RESULTS_DIR / "custom_eval"
CUSTOM_EVAL_MANIFEST_PATH = CUSTOM_EVAL_DIR / "zero_day_manifest.csv"
CUSTOM_EVAL_FAIL_IF_NOT_ENOUGH_BENIGN = True
CUSTOM_EVAL_FAIL_IF_NOT_ENOUGH_OLD_MALWARE = True
CUSTOM_EVAL_FAIL_IF_NEW_MALWARE_DIR_MISSING = True
CUSTOM_EVAL_ENFORCE_NEW_UNSEEN_DISJOINT = True
CUSTOM_EVAL_SHUFFLE_BEFORE_SAMPLE = True



OVERWRITE_ORIGINAL_METADATA = False