"""
pipeline_config_v2.py

Configuration for the new parallel raw-report -> graph -> enriched-graph pipeline.

Purpose
-------
This file isolates the new pipeline from the current one.

It defines:
- where the new raw reports live
- where raw/intermediate/enriched outputs should be written
- source names for schema-aware training
- activity-regime thresholds
- timestamp fallback policy
- graph-construction rules
- feature-dimension contracts that must stay compatible with the current model

Design goals
------------
Keep from current setup:
- parallel graph sources
- schema-per-source workflow
- same API node dim (775)
- same temporal edge dim (14)

Improve over current setup:
- stable resource dims per node type
- explicit low-activity handling
- explicit temporal-order policy
- explicit network/registry extraction policy
"""

from __future__ import annotations

from pathlib import Path

# ============================================================
# VERSION / SOURCE IDENTITY
# ============================================================
PIPELINE_VERSION = "v2"
RAW_SOURCE_NAME = "cuckoo_raw_v2"
ENRICHED_SOURCE_NAME = "cuckoo_enriched_v2"

# ============================================================
# ROOTS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
DEBUG_MAX_REPORTS = 5
# ============================================================
# INPUTS
# ============================================================
# Folder containing new raw sandbox JSON reports
RAW_REPORTS_DIR = DATA_DIR / "new_reports"

# Optional metadata / labels file if you have one
RAW_LABELS_CSV = DATA_DIR / "new_reports_labels.csv"

# Optional split files for the new source
SPLITS_DIR = DATA_DIR / "splits"

# ============================================================
# OUTPUTS
# ============================================================
# Canonical adapted report cache (optional but useful for debugging)
ADAPTED_REPORTS_DIR = CACHE_DIR / "adapted_reports_v2"

# Raw graph cache before enrichment
RAW_GRAPHS_DIR = CACHE_DIR / "graphs_v2"

# Final enriched graph cache used by schema/training
ENRICHED_GRAPHS_DIR = CACHE_DIR / ENRICHED_SOURCE_NAME

# ------------------------------------------------------------
# ZERO-DAY OUTPUT WIRING
# ------------------------------------------------------------
# Final graphs are written straight into the location the model's zero-day
# eval reads from (main config.CUSTOM_EVAL_NEW_MALWARE_DIR ==
# cache/zero_day_unseen/<variant>), so no manual move step is needed.
# Flip ZERO_DAY_VARIANT to repopulate a different bucket.
ZERO_DAY_VARIANT = "all"  # which zero-day bucket to populate: "all" | "old"
ZERO_DAY_OUTPUT_DIR = CACHE_DIR / "zero_day_unseen" / ZERO_DAY_VARIANT

# Save clean "<md5>.pkl" filenames (strip the "<report_folder>__" prefix) so
# names match metadata md5s and the eval's disjoint-leakage check works.
STRIP_REPORT_PREFIX_ON_SAVE = True

# Strip v2-only build-time attributes (pids, arguments_text, categories,
# statuses, time_sources, timestamps, process/network metadata, global meta)
# from the saved graph so its attribute set EXACTLY matches the training graphs
# (api: x/api_names/num_nodes; resources: x/num_nodes; edges: edge_index/
# edge_attr). Required — otherwise PyG cannot batch new graphs together with
# train/val/test/old graphs (KeyError on the extra attrs).
CONFORM_TO_TRAINING_SCHEMA = True

# Skip low-activity / failed-detonation reports (graphs with <= this many api
# nodes). These captured essentially no behavior (failed exec, evasion, short
# run), so they carry no malware signal and just read benign in the eval —
# they aren't valid zero-day detection cases. Skipping them at BUILD time means
# they are never produced again, and the run continues to the next report to
# still reach MAX_SUCCESSFUL_GRAPHS of real-activity graphs. Excluded md5s are
# logged to LOW_ACTIVITY_EXCLUDED_PATH.
SKIP_LOW_ACTIVITY_GRAPHS = True
MIN_API_NODES_TO_BUILD = 10  # skip if api node count <= this
LOW_ACTIVITY_EXCLUDED_PATH = DATA_DIR / "low_activity_excluded_md5s.csv"

# Schema output for the new source
SCHEMA_DIR = RESULTS_DIR / "schema" / ENRICHED_SOURCE_NAME
SCHEMA_JSON_PATH = SCHEMA_DIR / "schema.json"
SCHEMA_SUMMARY_PATH = SCHEMA_DIR / "schema_summary.txt"

# Logs / audits for the new pipeline
PIPELINE_LOG_DIR = RESULTS_DIR / "pipeline_logs" / ENRICHED_SOURCE_NAME
PIPELINE_AUDIT_DIR = RESULTS_DIR / "pipeline_audits" / ENRICHED_SOURCE_NAME

for path in [
    CACHE_DIR,
    RESULTS_DIR,
    ADAPTED_REPORTS_DIR,
    RAW_GRAPHS_DIR,
    ENRICHED_GRAPHS_DIR,
    ZERO_DAY_OUTPUT_DIR,
    SCHEMA_DIR,
    PIPELINE_LOG_DIR,
    PIPELINE_AUDIT_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

# ============================================================
# GRAPH SOURCE REGISTRATION
# ============================================================
# Add this source to the existing config.GRAPH_SOURCES later.
# This file only declares the intended source name/path for v2.
GRAPH_SOURCE_NAME = ENRICHED_SOURCE_NAME
GRAPH_SOURCE_PATH = ENRICHED_GRAPHS_DIR

# ============================================================
# REPORT VALIDATION / REGIME TAGGING
# ============================================================
# Keep all reports, but tag their regime explicitly.
REPORT_REGIMES = ("rich", "low_activity", "very_sparse", "invalid")

# Minimum evidence checks
MIN_PROCESSES_FOR_VALID = 1
MIN_PROCESS_TREE_NODES_FOR_VALID = 1

# Activity scoring thresholds
MIN_API_CALLS_RICH = 50
MIN_API_CALLS_LOW_ACTIVITY = 1

MIN_RESOURCE_EVENTS_RICH = 5
MIN_RESOURCE_EVENTS_LOW_ACTIVITY = 1

# If everything is basically absent, still keep the report but mark it.
ALLOW_VERY_SPARSE_REPORTS = True

# Whether to skip clearly broken reports from graph creation
SKIP_INVALID_REPORTS = False

# ============================================================
# TEMPORAL BACKBONE POLICY
# ============================================================
PRIMARY_TEMPORAL_NODE_TYPE = "api"

# Timestamp extraction policy
# Priority:
# 1. per-call time
# 2. process first_seen + monotonic index
# 3. pure event_index
USE_CALL_TIME_IF_PRESENT = True
USE_PROCESS_FIRST_SEEN_FALLBACK = True
USE_EVENT_INDEX_FALLBACK = True

# Temporal ordering must be deterministic
SORT_API_CALLS_BY = ("time", "event_index")
BURST_DELTA_THRESHOLD = 0.1
# IMPORTANT:
# temporal edges should be built within each process stream first
TEMPORAL_WITHIN_PROCESS_ONLY = True

# Whether to add any cross-process temporal links
ENABLE_CROSS_PROCESS_TEMPORAL_EDGES = False

# Max API calls per graph (must stay aligned with current training assumptions)
MAX_API_CALLS = 10000
MIN_API_CALLS = 0  # keep sparse reports too; regime tags will distinguish them

# ============================================================
# NODE / EDGE TYPE CONTRACT
# ============================================================
NODE_TYPES = ["api", "file", "registry", "process", "network"]

EDGE_TYPES = [
    ("api", "temporal", "api"),
    ("api", "operates_on", "file"),
    ("api", "modifies", "registry"),
    ("process", "executes", "api"),
    ("api", "connects_to", "network"),
]

DEFAULT_NODE_TYPE_ORDER = ["api", "file", "registry", "process", "network"]

DEFAULT_EDGE_TYPE_ORDER = [
    ("api", "temporal", "api"),
    ("api", "operates_on", "file"),
    ("api", "modifies", "registry"),
    ("process", "executes", "api"),
    ("api", "connects_to", "network"),
]

# ============================================================
# FEATURE-DIMENSION CONTRACT
# ============================================================
# Must remain compatible with current model/schema expectations.
API_NODE_DIM = 775
TEMPORAL_EDGE_DIM = 14

# New pipeline improvement:
# enforce stable dims per resource family instead of mixed dims.
FILE_NODE_DIM = 6
REGISTRY_NODE_DIM = 6
PROCESS_NODE_DIM = 6
NETWORK_NODE_DIM = 6

RESOURCE_NODE_DIMS = {
    "file": FILE_NODE_DIM,
    "registry": REGISTRY_NODE_DIM,
    "process": PROCESS_NODE_DIM,
    "network": NETWORK_NODE_DIM,
}

# ============================================================
# TRAINING-PATTERN API FEATURE ALIGNMENT
# ============================================================
# Build the 775-dim api features the SAME way the training (cape_enriched)
# graphs were built, so zero-day graphs land in the training distribution.
# CRITICAL: the kept checkpoint has NO model-side normalization to absorb a
# feature shift, so this must match training exactly.
#   - 768 semantic = NAME-ONLY embedding via the training api_embeddings.pkl
#     lookup (NOT v2's default rich "api=..|category=..|args=.." re-encode).
#   - 7 raw dims   = [time_norm, pid/10000, is_file, is_registry, is_process,
#                     is_network, is_crypto]  (name-substring category flags,
#                     matching data_processing/graph_builder.py:add_api_nodes).
MATCH_TRAINING_API_FEATURES = True
TRAINING_API_EMBEDDINGS_PATH = CACHE_DIR / "api_embeddings.pkl"
# Unknown (zero-day) API names not in the lookup:
#   True  = encode name-only with the same model (better for new APIs, keeps
#           them in the same semantic space; what the user asked for earlier).
#   False = zeros (strict training behaviour for unknown APIs).
ENCODE_UNKNOWN_APIS_BY_NAME = True

# ============================================================
# EDGE FEATURE POLICY
# ============================================================
# Only temporal edges carry edge attributes in the current design.
TEMPORAL_EDGE_REL = ("api", "temporal", "api")

# IMPORTANT:
# This 14-dim contract matches the previous edge enrichment logic.
# Keep this aligned with the old enriched pipeline unless you
# intentionally redesign both enrichment + schema expectations.
TEMPORAL_EDGE_FEATURE_NAMES = [
    "time_delta",            # 0  original/basic
    "same_process",          # 1  original/basic
    "is_burst",              # 2  original/basic
    "speed",                 # 3  new
    "acceleration",          # 4  new
    "burst_intensity",       # 5  new
    "local_burst_ratio",     # 6  new
    "time_from_start",       # 7  new
    "time_to_end",           # 8  new
    "sequence_position",     # 9  new
    "window_mean_dt",        # 10 new
    "window_std_dt",         # 11 new
    "periodicity_score",     # 12 new
    "sudden_change_flag",    # 13 new
]

assert len(TEMPORAL_EDGE_FEATURE_NAMES) == TEMPORAL_EDGE_DIM

# ============================================================
# RESOURCE EXTRACTION POLICY
# ============================================================
# File extraction
ENABLE_FILE_NODES = True
NORMALIZE_FILE_PATHS = True

# Registry extraction
ENABLE_REGISTRY_NODES = True
NORMALIZE_REGISTRY_KEYS = True

# Network extraction
ENABLE_NETWORK_NODES = True

# Be conservative with noisy background traffic
FILTER_BACKGROUND_NETWORK = True
BACKGROUND_IPS = {
    "224.0.0.252",
    "239.255.255.250",
}
BACKGROUND_HOSTS = {
    "time.windows.com",
    "dns.msftncsi.com",
    "teredo.ipv6.microsoft.com",
}

# ============================================================
# PROCESS POLICY
# ============================================================
# Use PID as primary join key, but allow path/first_seen disambiguation.
USE_PID_AS_PRIMARY_PROCESS_KEY = True
USE_PROCESS_PATH_FOR_DISAMBIGUATION = True
USE_FIRST_SEEN_FOR_DISAMBIGUATION = True

# ============================================================
# LOW-ACTIVITY / QUALITY FEATURES
# ============================================================
# These are graph-level or auxiliary structural features, not labels.
ATTACH_ACTIVITY_METADATA = True

GRAPH_ACTIVITY_FEATURE_NAMES = [
    "activity_score",
    "is_rich",
    "is_low_activity",
    "is_very_sparse",
    "num_api_calls",
    "num_process_nodes",
    "num_file_nodes",
    "num_registry_nodes",
    "num_network_nodes",
]

# ============================================================
# ENRICHMENT POLICY
# ============================================================
# Reuse current enrichment stack assumptions.
ENABLE_NODE_ENRICHMENT = True
ENABLE_EDGE_ENRICHMENT = True

# Reuse current LLM/API embedding flow
MAX_VOCAB_SIZE = 5000
VOCAB_SAVE_PATH = RESULTS_DIR / f"api_vocabulary_{ENRICHED_SOURCE_NAME}.json"

# Check vocabulary coverage before large runs
CHECK_API_VOCAB_COVERAGE = True
MIN_API_VOCAB_COVERAGE = 0.85

# ============================================================
# SERIALIZATION / CHUNKING
# ============================================================
CHUNK_SIZE = 100
SAVE_ADAPTED_REPORTS = False
SAVE_RAW_GRAPHS = False
SAVE_ENRICHED_GRAPHS = True

# ============================================================
# AUDIT / VALIDATION
# ============================================================
RUN_POST_BUILD_AUDIT = True
RUN_SCHEMA_INSPECTION_AFTER_BUILD = True
FAIL_ON_DIM_MISMATCH = True
FAIL_ON_MISSING_API_DIM = True
FAIL_ON_MISSING_TEMPORAL_EDGE_DIM = True

# ============================================================
# LABELS
# ============================================================
CLASS_NAMES = {
    0: "Ransomware",
    1: "Benign",
    2: "Other Malwares",
}

LABEL_NAME_TO_ID = {
    "ransomware": 0,
    "benign": 1,
    "malware":2,
    "other_malware": 2,
    "other": 2,
}

# ============================================================
# DEBUGGING
# ============================================================
DEBUG_MAX_REPORTS = None
DEBUG_PRINT_EVERY = 25
VERBOSE = True
DATA_METADATA_PATH = DATA_DIR / "extracted_metadata_appended.csv"
MAX_SUCCESSFUL_GRAPHS = 140