"""
PyTorch Dataset for Temporal Heterogeneous GNN (TGNN) Ransomware Detection

Preserves HeteroData structure with multiple node/edge types for temporal analysis.

3-CLASS CLASSIFICATION:
    - 0: Ransomware
    - 1: Benign
    - 2: Other Malware (trojan, banker, adware, etc.)
"""

from __future__ import annotations

import copy
import logging
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, HeteroData
from config import METADATA_PATH
import config
from utils.schema_manager import ensure_active_schema
from utils.schema_utils import (
    get_active_schema,
    get_canonical_node_dims,
    get_node_types,
    normalize_graph_node_features,
    summarize_schema,
)
from utils.schema_validator import (
    validate_batch_or_raise,
    validate_graph_or_raise,
)
import numpy as np
from torch.utils.data import Sampler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Size-Aware Batch Sampler
# ══════════════════════════════════════════════════════════════════════════════

class SizeAwareBatchSampler(Sampler):
    """
    Each batch = 1 large graph + (batch_size-1) small graphs.
    Maximally dilutes monster graphs — eliminates OOM entirely.
    Uses file size on disk as proxy (no graph loading at startup).
    Optionally accepts per-sample weights for weighted oversampling.
    """
    def __init__(self, dataset, batch_size, shuffle=True, weights=None):
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.weights    = np.array(weights) if weights is not None else None

        # ── FAST: file size on disk, zero graph loading ───────────────────
        print("📏 Measuring graph sizes (fast, disk-only)...")
        sizes = np.array([
            f.stat().st_size for f in dataset.graph_files
        ], dtype=np.float32)

        self.sizes   = sizes
        self.indices = np.argsort(sizes)   # sorted: smallest → largest

        # top (1/batch_size) are "large", rest are "small"
        # e.g. batch_size=6 → top 17% large, bottom 83% small
        n_large      = max(1, len(self.indices) // batch_size)
        self.small_i = self.indices[:-n_large].tolist()
        self.large_i = self.indices[-n_large:].tolist()

        print(f"   Smallest     : {sizes[self.indices[0]]  / 1024:.1f} KB")
        print(f"   Largest      : {sizes[self.indices[-1]] / 1024 / 1024:.1f} MB")
        print(f"   Large bucket : {len(self.large_i)} graphs (top {100 // batch_size}%)")
        print(f"   Small bucket : {len(self.small_i)} graphs (bot {100 - 100 // batch_size}%)")
        print(f"   Batches      : {len(self):,}")

    def __iter__(self):
        small_i           = self.small_i.copy()
        large_i           = self.large_i.copy()
        n_small_per_batch = self.batch_size - 1   # 5 small + 1 large per batch
        n_batches         = len(large_i)

        if self.weights is not None:
            # ── weighted path ─────────────────────────────────────────────
            small_w = self.weights[small_i]
            large_w = self.weights[large_i]
            small_w = small_w / small_w.sum()
            large_w = large_w / large_w.sum()

            small_pool = np.random.choice(
                small_i, size=n_batches * n_small_per_batch,
                p=small_w, replace=True
            ).tolist()
            large_pool = np.random.choice(
                large_i, size=n_batches,
                p=large_w, replace=True
            ).tolist()
        else:
            # ── standard path ────────────────────────────────────────────
            if self.shuffle:
                np.random.shuffle(small_i)
                np.random.shuffle(large_i)
            # tile small pool if needed to fill all batches
            repeats    = (n_batches * n_small_per_batch // len(small_i)) + 1
            small_pool = (small_i * repeats)[: n_batches * n_small_per_batch]
            large_pool = large_i[:n_batches]

        # build batches: [s, s, s, s, s, L]
        batches = [
            small_pool[i * n_small_per_batch : (i + 1) * n_small_per_batch] + [large_pool[i]]
            for i in range(n_batches)
        ]

        if self.shuffle:
            np.random.shuffle(batches)

        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.large_i)


# ══════════════════════════════════════════════════════════════════════════════
# Class name mapping
# ══════════════════════════════════════════════════════════════════════════════

CLASS_NAMES = {
    0: "Ransomware",
    1: "Benign",
    2: "Other Malware",
}


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class RansomwareGraphDataset(Dataset):
    """
    Dataset for Temporal Heterogeneous Graphs.

    Each HeteroData graph contains:
        Node types: ['api', 'file', 'registry', 'process', 'network']
        Edge types: heterogeneous relations based on active schema

    Features:
        - API nodes: expected enriched schema feature space
        - Temporal edge features: expected enriched schema feature space
        - Label: 0=Ransomware, 1=Benign, 2=Other Malware
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        cache_dir: Optional[str] = None,
        schema: Optional[Dict] = None,
        validate_graphs: bool = True,
    ):
        self.data_dir  = Path(data_dir)
        self.split     = split
        self.cache_dir = Path(cache_dir) if cache_dir is not None else config.ACTIVE_GRAPH_DIR
        self.split_dir = self.cache_dir / split
        self.enable_augmentation = (split == "train")
        self.validate_graphs     = validate_graphs

        self.schema             = schema if schema is not None else ensure_active_schema()
        self.canonical_node_dims = get_canonical_node_dims(self.schema)
        self.node_type_order    = get_node_types(self.schema)

        # Load graph file paths
        self.graph_files = sorted(list(self.split_dir.glob("*.pkl")))

        if len(self.graph_files) == 0:
            raise ValueError(f"No .pkl files found in {self.split_dir}")

        logger.info(f"Loaded {len(self.graph_files)} HeteroData graphs for {split} split")
        logger.info(f"Using graph root: {self.cache_dir}")

        # Load labels
        self.labels = self._load_labels()

    def _load_labels(self) -> Dict[str, int]:
        """Load 3-class labels from metadata CSV."""
        metadata_path = Path(METADATA_PATH)

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        df = pd.read_csv(metadata_path)
        labels = {}
        for _, row in df.iterrows():
            sample_id = str(row["id"])
            label_str = str(row.get("label", "")).lower().strip()

            if label_str in ("ransomware", "0"):
                labels[sample_id] = 0
            elif label_str in ("benign", "1"):
                labels[sample_id] = 1
            else:
                labels[sample_id] = 2   # other malware

        # Log split distribution
        split_counts = {0: 0, 1: 0, 2: 0}
        for f in self.graph_files:
            sid = f.stem
            lbl = labels.get(sid, 2)
            split_counts[lbl] += 1

        logger.info(f"  {self.split} split:")
        logger.info(f"    - Ransomware: {split_counts[0]}")
        logger.info(f"    - Benign: {split_counts[1]}")
        logger.info(f"    - Other malware: {split_counts[2]}")

        return labels

    def __len__(self) -> int:
        return len(self.graph_files)

    def __getitem__(self, idx: int) -> Tuple[HeteroData, int]:
        graph_path = self.graph_files[idx]
        sample_id  = graph_path.stem

        with open(graph_path, "rb") as f:
            data = pickle.load(f)

        label = self.labels.get(sample_id, 2)

        if self.enable_augmentation and config.USE_AUGMENTATION:
            data = self._augment(data)

        return data, label

    def _augment(self,  HeteroData) -> HeteroData:
        """Light augmentation: feature noise + dropout + edge dropout."""
        data = copy.deepcopy(data)

        # 1) Feature noise + feature dropout on all node types
        for nt in self.node_type_order:
            if nt in data.node_types and hasattr(data[nt], "x"):
                x = data[nt].x
                if isinstance(x, torch.Tensor) and x.numel() > 0:
                    noise = 0.01 * torch.randn_like(x)
                    x     = x + noise
                    mask  = (torch.rand_like(x) > 0.05).float()
                    x     = x * mask
                    data[nt].x = x

        # 2) Light dropout on non-temporal edges only
        edge_types_to_drop = [
            ("api", "operates_on", "file"),
            ("api", "modifies", "registry"),
            ("api", "communicates_with", "network"),
            ("api", "spawns", "process"),
        ]
        for et in edge_types_to_drop:
            if et in data.edge_types and hasattr(data[et], "edge_index"):
                ei = data[et].edge_index
                if ei is not None and ei.numel() > 0:
                    keep_mask = torch.rand(ei.shape[1]) > 0.05
                    data[et].edge_index = ei[:, keep_mask]
                    if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None:
                        data[et].edge_attr = data[et].edge_attr[keep_mask]

        return data


# ══════════════════════════════════════════════════════════════════════════════
# Graph cleaning helper
# ══════════════════════════════════════════════════════════════════════════════

def _clean_graph_for_batching(graph: HeteroData) -> Optional[HeteroData]:
    """
    Ensure all node types in the schema exist in the graph.
    Missing node types get an empty placeholder tensor.
    Returns None if the graph is fundamentally broken.
    """
    try:
        schema = get_active_schema()
        canonical_dims = get_canonical_node_dims(schema)

        for nt, dim in canonical_dims.items():
            if nt not in graph.node_types or not hasattr(graph[nt], "x") or graph[nt].x is None:
                graph[nt].x = torch.zeros((0, dim), dtype=torch.float32)

        return graph

    except Exception as e:
        logger.warning(f"_clean_graph_for_batching failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Collate function
# ══════════════════════════════════════════════════════════════════════════════

def hetero_collate(
    batch,
    schema: Optional[Dict] = None,
    validate_batch: bool = False,
) -> Tuple[Optional[Batch], Optional[torch.Tensor]]:
    """
    Collate a list of (HeteroData, label) tuples into a batched graph + label tensor.
    Returns (None, None) if the batch cannot be assembled.
    """
    if not batch:
        logger.warning("hetero_collate received empty batch")
        return None, None

    if schema is None:
        schema = get_active_schema()

    graphs, labels = zip(*batch)
    labels_tensor  = torch.tensor(labels, dtype=torch.long)

    processed_graphs = []
    for graph in graphs:
        # Normalize feature dims to match canonical schema
        if config.NORMALIZE_NODE_FEATURE_DIMS:
            graph = normalize_graph_node_features(graph, schema=schema, inplace=True)

        # Optional per-graph validation (disabled in training for speed)
        if validate_batch and config.FAIL_ON_SCHEMA_MISMATCH:
            validate_graph_or_raise(
                graph,
                schema=schema,
                require_api_nodes=True,
                strict=config.SCHEMA_STRICT_VALIDATION,
            )

        graph = _clean_graph_for_batching(graph)

        if graph is None:
            logger.warning("_clean_graph_for_batching returned None — skipping graph")
            continue   # ← drop silently, never append None

        processed_graphs.append(graph)

    # Guard: entire batch was dropped
    if not processed_graphs:
        logger.warning("hetero_collate: all graphs failed cleaning — skipping batch")
        return None, None

    batched_graph = Batch.from_data_list(processed_graphs)

    # Optional full-batch schema validation (disabled in training for speed)
    if validate_batch and config.FAIL_ON_SCHEMA_MISMATCH:
        validate_batch_or_raise(
            batched_graph,
            schema=schema,
            strict=config.SCHEMA_STRICT_VALIDATION,
        )

    return batched_graph, labels_tensor


# ══════════════════════════════════════════════════════════════════════════════
# Dataset validation utility
# ══════════════════════════════════════════════════════════════════════════════

def validate_dataset(data_dir: str = "data", cache_dir: Optional[str] = None):
    """Validate HeteroData graphs under the active schema."""
    logger.info("\n" + "=" * 70)
    logger.info("DATASET VALIDATION")
    logger.info("=" * 70)

    schema = get_active_schema()
    logger.info(f"\nActive schema summary:\n{summarize_schema(schema)}")

    for split in ["train", "val", "test"]:
        try:
            dataset = RansomwareGraphDataset(
                data_dir=data_dir,
                split=split,
                cache_dir=cache_dir,
                schema=schema,
                validate_graphs=True,
            )
            logger.info(f"\n{split}: {len(dataset)} graphs")
        except Exception as e:
            logger.error(f"Error loading {split} split: {e}")
