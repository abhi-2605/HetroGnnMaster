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



CLASS_NAMES = {
    0: "Ransomware",
    1: "Benign",
    2: "Other Malware",
}


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
        self.data_dir = Path(data_dir)
        self.split = split
        self.cache_dir = Path(cache_dir) if cache_dir is not None else config.ACTIVE_GRAPH_DIR
        self.split_dir = self.cache_dir / split
        self.enable_augmentation = (split == "train")
        self.validate_graphs = validate_graphs

        # Ensure active schema is present and current
        self.schema = schema if schema is not None else ensure_active_schema()
        self.canonical_node_dims = get_canonical_node_dims(self.schema)
        self.node_type_order = get_node_types(self.schema)

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
            hash_id = row["md5"]
            category_names = str(row["all_categories"]).lower()

            if "ransomware" in category_names:
                label = 0  # Ransomware
            elif "benign" in category_names:
                label = 1  # Benign
            else:
                label = 2  # Other malware

            labels[hash_id] = label

        logger.info(f"Loaded {len(labels)} labels from metadata CSV")

        split_labels = []
        for graph_file in self.graph_files:
            hash_id = graph_file.stem
            if hash_id in labels:
                split_labels.append(labels[hash_id])

        logger.info(f"  {self.split} split:")
        logger.info(f"    - Ransomware: {sum(1 for l in split_labels if l == 0)}")
        logger.info(f"    - Benign: {sum(1 for l in split_labels if l == 1)}")
        logger.info(f"    - Other malware: {sum(1 for l in split_labels if l == 2)}")

        return labels

    def __len__(self) -> int:
        return len(self.graph_files)
    
    def _ensure_api_timestamps(self, data: HeteroData) -> HeteroData:
        """
        Ensure api.timestamps exists.

        If missing, reconstruct relative timestamps from the temporal api->api
        edges using edge_attr[:, 0] as time_delta. This is only a fallback and
        is used only when explicit api.timestamps are absent.
        """
        if "api" not in data.node_types:
            return data

        # Already present and aligned -> keep as-is
        if hasattr(data["api"], "timestamps"):
            ts = data["api"].timestamps
            if ts is not None and hasattr(data["api"], "x"):
                if len(ts) == data["api"].x.shape[0]:
                    return data

        edge_type = ("api", "temporal", "api")
        if edge_type not in data.edge_types:
            return data

        store = data[edge_type]
        if not hasattr(store, "edge_index") or store.edge_index is None:
            return data
        if not hasattr(store, "edge_attr") or store.edge_attr is None:
            return data
        if store.edge_attr.ndim != 2 or store.edge_attr.shape[1] < 1:
            return data

        if not hasattr(data["api"], "x"):
            return data

        num_api = int(data["api"].x.shape[0])
        if num_api == 0:
            data["api"].timestamps = torch.empty(0, dtype=torch.float32)
            return data

        edge_index = store.edge_index
        edge_attr = store.edge_attr

        src = edge_index[0].detach().cpu().tolist()
        dst = edge_index[1].detach().cpu().tolist()
        deltas = edge_attr[:, 0].detach().cpu().float().tolist()

        incoming = {}
        outgoing = {}
        for s, d, dt in zip(src, dst, deltas):
            outgoing.setdefault(s, []).append((d, dt))
            if d not in incoming:
                incoming[d] = (s, dt)

        # Nodes with no incoming temporal edge are treated as starts
        start_nodes = [i for i in range(num_api) if i not in incoming]
        if not start_nodes:
            start_nodes = [0]

        timestamps = [None] * num_api

        # Initialize start nodes at 0.0
        for s in start_nodes:
            timestamps[s] = 0.0

        # Propagate cumulative deltas forward
        stack = list(start_nodes)
        while stack:
            u = stack.pop()
            base_t = timestamps[u]
            for v, dt in outgoing.get(u, []):
                cand = float(base_t) + float(dt)
                if timestamps[v] is None or cand < timestamps[v]:
                    timestamps[v] = cand
                    stack.append(v)

        # Fill disconnected/unreached nodes safely
        last_t = 0.0
        for i in range(num_api):
            if timestamps[i] is None:
                timestamps[i] = last_t
            last_t = timestamps[i]

        data["api"].timestamps = torch.tensor(timestamps, dtype=torch.float32)

        # Optional provenance marker
        try:
            data["api"].time_sources = ["temporal_edge_fallback"] * num_api
        except Exception:
            pass

        return data

    def __getitem__(self, idx: int) -> Tuple[HeteroData, int]:
        """
        Load HeteroData graph with temporal structure preserved.

        Returns:
            Tuple of (HeteroData graph, label)
        """
        graph_path = self.graph_files[idx]

        with open(graph_path, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, HeteroData):
            raise TypeError(f"Expected HeteroData, got {type(data)}")

        # Fallback: reconstruct api.timestamps from temporal edge deltas if missing
        data = self._ensure_api_timestamps(data)

        hash_id = graph_path.stem
        label = self.labels.get(hash_id, 2)
        if hash_id not in self.labels:
            logger.warning(f"No label found for {hash_id}, defaulting to 2")

        # Normalize to active schema dims
        if config.NORMALIZE_NODE_FEATURE_DIMS:
            data = normalize_graph_node_features(data, schema=self.schema, inplace=True)

        # Validate normalized graph
        if self.validate_graphs and config.FAIL_ON_SCHEMA_MISMATCH:
            validate_graph_or_raise(
                data,
                schema=self.schema,
                require_api_nodes=True,
                strict=config.SCHEMA_STRICT_VALIDATION,
            )

        # Train-time augmentation for benign graphs only
        if self.enable_augmentation and label == 1:
            if random.random() < 0.7:
                data = self.augment_benign_graph(data)

                # Re-normalize after augmentation just to be safe
                if config.NORMALIZE_NODE_FEATURE_DIMS:
                    data = normalize_graph_node_features(data, schema=self.schema, inplace=True)

        return data, label

    def augment_benign_graph(self, data: HeteroData) -> HeteroData:
        """
        Light augmentation for benign training graphs.
        Designed to reduce memorization without destroying semantics.
        """
        data = copy.deepcopy(data)

        resource_node_types = ["file", "registry", "process", "network"]

        # 1) Small Gaussian noise + light feature masking on resource nodes
        for nt in resource_node_types:
            if nt in data.node_types and hasattr(data[nt], "x"):
                x = data[nt].x
                if isinstance(x, torch.Tensor) and x.numel() > 0:
                    noise = 0.01 * torch.randn_like(x)
                    x = x + noise

                    mask = (torch.rand_like(x) > 0.05).float()
                    x = x * mask

                    data[nt].x = x

        # 2) Light dropout on non-temporal edges only
        edge_types_to_drop = [
            ("api", "operates_on", "file"),
            ("api", "modifies", "registry"),
            ("process", "executes", "api"),
            ("api", "connects_to", "network"),
        ]

        for et in edge_types_to_drop:
            if et in data.edge_types:
                store = data[et]
                if hasattr(store, "edge_index") and isinstance(store.edge_index, torch.Tensor):
                    E = store.edge_index.size(1)
                    if E > 0:
                        keep_mask = torch.rand(E, device=store.edge_index.device) > 0.08
                        if keep_mask.sum() == 0:
                            keep_mask[torch.randint(0, E, (1,), device=keep_mask.device)] = True

                        store.edge_index = store.edge_index[:, keep_mask]

                        if hasattr(store, "edge_attr") and isinstance(store.edge_attr, torch.Tensor):
                            if store.edge_attr.size(0) == E:
                                store.edge_attr = store.edge_attr[keep_mask]

        return data

    def get_statistics(self) -> dict:
        """Get dataset statistics for HeteroData."""
        split_labels = []
        for graph_file in self.graph_files:
            hash_id = graph_file.stem
            if hash_id in self.labels:
                split_labels.append(self.labels[hash_id])

        stats = {
            "num_graphs": len(self),
            "num_ransomware": sum(1 for l in split_labels if l == 0),
            "num_benign": sum(1 for l in split_labels if l == 1),
            "num_other_malware": sum(1 for l in split_labels if l == 2),
            "split": self.split,
            "graph_type": "HeteroData (Temporal)",
            "graph_root": str(self.cache_dir),
        }

        if len(self.graph_files) > 0:
            with open(self.graph_files[0], "rb") as f:
                sample = pickle.load(f)

            if isinstance(sample, HeteroData):
                if config.NORMALIZE_NODE_FEATURE_DIMS:
                    sample = normalize_graph_node_features(sample, schema=self.schema, inplace=True)

                stats["node_types"] = sample.node_types
                stats["edge_types"] = sample.edge_types

                for node_type in sample.node_types:
                    if hasattr(sample[node_type], "x"):
                        stats[f"{node_type}_nodes"] = sample[node_type].x.shape[0]
                        stats[f"{node_type}_features"] = sample[node_type].x.shape[1]

        return stats


def _clean_graph_for_batching(graph: HeteroData) -> HeteroData:
    keep_graph_level = set()
    keep_store_attrs = {"timestamps", "timestamp", "time", "num_nodes"}
    drop_store_attrs = {"placeholder_count"}   # explicitly drop known bad extras

    # Clean graph-level attrs: remove everything not explicitly allowed
    top_keys_to_remove = []
    for key in list(graph.keys()):
        if key not in keep_graph_level:
            top_keys_to_remove.append(key)

    for key in top_keys_to_remove:
        try:
            del graph[key]
        except Exception:
            pass

    # Clean node stores
    for node_type in graph.node_types:
        store = graph[node_type]
        keys_to_remove = []
        for key in list(store.keys()):
            if key in drop_store_attrs:
                keys_to_remove.append(key)
                continue
            if key in keep_store_attrs:
                continue
            try:
                val = store[key]
                if not isinstance(val, torch.Tensor):
                    keys_to_remove.append(key)
            except Exception:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            try:
                del store[key]
            except Exception:
                pass

    # Clean edge stores
    for edge_type in graph.edge_types:
        store = graph[edge_type]
        keys_to_remove = []
        for key in list(store.keys()):
            if key in keep_store_attrs:
                continue
            try:
                val = store[key]
                if not isinstance(val, torch.Tensor):
                    keys_to_remove.append(key)
            except Exception:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            try:
                del store[key]
            except Exception:
                pass

    for node_type in graph.node_types:
        if hasattr(graph[node_type], "x"):
            graph[node_type].num_nodes = graph[node_type].x.shape[0]

    return graph

def _dynamic_cap_api_nodes(graphs: list, safety_fraction: float = 0.75) -> list:
    """
    Dynamically cap API node counts so the batch fits in available GPU memory.
    Falls back to original graphs untouched if anything goes wrong.
    """
    try:
        import torch

        if not torch.cuda.is_available() or not graphs:
            return graphs

        free_bytes         = torch.cuda.mem_get_info()[0]
        safe_bytes         = free_bytes * safety_fraction
        bytes_per_api_node = 8 * 1024
        total_node_budget  = int(safe_bytes / bytes_per_api_node)

        node_counts = []
        for g in graphs:
            try:
                n = g['api'].x.shape[0] if (hasattr(g['api'], 'x') and g['api'].x is not None) else 0
            except Exception:
                n = 0
            node_counts.append(n)

        total_nodes = sum(node_counts)

        # ── fits in budget — touch nothing ───────────────────────────────
        if total_nodes <= total_node_budget:
            return graphs

        trimmed = []
        for g, n in zip(graphs, node_counts):
            try:
                if n == 0:
                    trimmed.append(g)
                    continue

                fair_share = max(512, int(total_node_budget * n / total_nodes))

                if n <= fair_share:
                    trimmed.append(g)   # already fits — untouched ✅
                    continue

                # subsample, preserve temporal order
                perm    = torch.randperm(n)[:fair_share]
                perm, _ = perm.sort()

                g['api'].x = g['api'].x[perm]

                if hasattr(g['api'], 'timestamps') and g['api'].timestamps is not None:
                    g['api'].timestamps = g['api'].timestamps[perm]

                # remap api→api edges
                old2new = torch.full((n,), -1, dtype=torch.long)
                old2new[perm] = torch.arange(fair_share)

                for et in list(g.edge_types):
                    src, _, dst = et
                    if src == 'api' and dst == 'api':
                        try:
                            ei   = g[et].edge_index
                            mask = (old2new[ei[0]] >= 0) & (old2new[ei[1]] >= 0)
                            g[et].edge_index = old2new[ei[:, mask]]
                            if hasattr(g[et], 'edge_attr') and g[et].edge_attr is not None:
                                g[et].edge_attr = g[et].edge_attr[mask]
                        except Exception:
                            pass   # edge remap failed — keep original edges

                trimmed.append(g)

            except Exception:
                trimmed.append(g)   # this graph failed — keep it as-is

        # safety: never return fewer graphs than we received
        if len(trimmed) != len(graphs):
            return graphs

        trimmed_total = sum(
            g['api'].x.shape[0]
            if (hasattr(g['api'], 'x') and g['api'].x is not None) else 0
            for g in trimmed
        )
        if total_nodes != trimmed_total:
            print(f"   ✂️  API nodes: {total_nodes:,} → {trimmed_total:,} "
                  f"(budget {total_node_budget:,}, free {free_bytes/1e9:.1f} GB)")

        return trimmed

    except Exception as e:
        print(f"⚠️  _dynamic_cap_api_nodes failed ({e}) — using original graphs")
        return graphs   # ← always falls back, never crashes the training loop


def hetero_collate(
    batch,
    schema: Optional[Dict] = None,
    validate_batch: bool = True,
):
    """
    Collate function for HeteroData batching with schema-aware normalization and validation.
    """
    schema = schema if schema is not None else get_active_schema()

    graphs = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)

    # Preserve API names for vocabulary / memory indexing
    all_api_names = []
    for graph in graphs:
        if "api" in graph.node_types and hasattr(graph["api"], "api_names"):
            all_api_names.extend(graph["api"].api_names)

    # Collect timestamps from all graphs before cleaning
    timestamps_dict = {}
    if len(graphs) > 0:
        all_node_types = set()
        for graph in graphs:
            all_node_types.update(graph.node_types)

        for node_type in all_node_types:
            all_timestamps = []
            for graph in graphs:
                if node_type in graph.node_types and hasattr(graph[node_type], "timestamps"):
                    all_timestamps.append(graph[node_type].timestamps)
            if all_timestamps:
                timestamps_dict[node_type] = torch.cat(all_timestamps)

    processed_graphs = []
    for graph in graphs:
        # Make sure dimensions match canonical schema
        if config.NORMALIZE_NODE_FEATURE_DIMS:
            graph = normalize_graph_node_features(graph, schema=schema, inplace=True)

        # Validate per-graph before batching
        if validate_batch and config.FAIL_ON_SCHEMA_MISMATCH:
            validate_graph_or_raise(
                graph,
                schema=schema,
                require_api_nodes=True,
                strict=config.SCHEMA_STRICT_VALIDATION,
            )

        graph = _clean_graph_for_batching(graph)
        processed_graphs = _dynamic_cap_api_nodes(processed_graphs)

    # Batch graphs
    batched_graph = Batch.from_data_list(processed_graphs)

    # Restore auxiliary attributes needed later
    if all_api_names:
        batched_graph["api"].api_names = all_api_names

    batched_graph.graph_timestamps = timestamps_dict if timestamps_dict else None

    # Build model-facing dictionaries for validation
    x_dict = {}
    edge_index_dict = {}
    edge_attr_dict = {}
    batch_dict = {}

    for node_type in batched_graph.node_types:
        if hasattr(batched_graph[node_type], "x"):
            x_dict[node_type] = batched_graph[node_type].x
        if hasattr(batched_graph[node_type], "batch"):
            batch_dict[node_type] = batched_graph[node_type].batch

    for edge_type in batched_graph.edge_types:
        if hasattr(batched_graph[edge_type], "edge_index"):
            edge_index_dict[edge_type] = batched_graph[edge_type].edge_index
        if hasattr(batched_graph[edge_type], "edge_attr"):
            edge_attr_dict[edge_type] = batched_graph[edge_type].edge_attr

    if validate_batch and config.FAIL_ON_SCHEMA_MISMATCH:
        validate_batch_or_raise(
            x_dict=x_dict,
            edge_index_dict=edge_index_dict,
            edge_attr_dict=edge_attr_dict if edge_attr_dict else None,
            batch_dict=batch_dict if batch_dict else None,
            schema=schema,
            require_api=True,
            require_api_indices_if_memory=False,  # api_indices are created later in train.py
            api_indices=None,
        )

    return batched_graph, labels


def create_dataloaders(
    data_dir: str = "data",
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle_train: bool = True,
    schema: Optional[Dict] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for HeteroData graphs.

    Uses active schema-aware collate to handle graph-level labels separately.
    """
    try:
        schema = schema if schema is not None else ensure_active_schema()

        # Create datasets
        train_dataset = RansomwareGraphDataset(
            data_dir=data_dir,
            split="train",
            cache_dir=cache_dir,
            schema=schema,
            validate_graphs=True,
        )
        val_dataset = RansomwareGraphDataset(
            data_dir=data_dir,
            split="val",
            cache_dir=cache_dir,
            schema=schema,
            validate_graphs=True,
        )
        test_dataset = RansomwareGraphDataset(
            data_dir=data_dir,
            split="test",
            cache_dir=cache_dir,
            schema=schema,
            validate_graphs=True,
        )

        schema_summary = summarize_schema(schema)
        logger.info("\nActive schema summary:")
        logger.info(f"  Source: {schema_summary['source_name']}")
        logger.info(f"  Graph root: {schema_summary['graph_root']}")
        logger.info(f"  Node types: {schema_summary['node_types']}")
        logger.info(f"  Edge types: {schema_summary['edge_types']}")
        logger.info(f"  Canonical node dims: {schema_summary['canonical_node_dims']}")

        for dataset in [train_dataset, val_dataset, test_dataset]:
            stats = dataset.get_statistics()
            logger.info(f"\n{stats['split'].upper()} Split (HeteroData):")
            logger.info(f"  Graphs: {stats['num_graphs']}")
            logger.info(f"  Ransomware: {stats['num_ransomware']}")
            logger.info(f"  Benign: {stats['num_benign']}")
            logger.info(f"  Other malware: {stats['num_other_malware']}")
            if "node_types" in stats:
                logger.info(f"  Node types: {stats['node_types']}")
                logger.info(f"  Edge types: {len(stats['edge_types'])} types")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            collate_fn=lambda batch: hetero_collate(batch, schema=schema, validate_batch=True),
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda batch: hetero_collate(batch, schema=schema, validate_batch=True),
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda batch: hetero_collate(batch, schema=schema, validate_batch=True),
        )

        return train_loader, val_loader, test_loader

    except Exception as e:
        logger.error(f"Error creating dataloaders: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def validate_dataset(data_dir: str = "data", cache_dir: Optional[str] = None):
    """Validate HeteroData graphs under the active schema."""
    logger.info("\n" + "=" * 70)
    logger.info("Temporal HeteroData Graph Validation")
    logger.info("=" * 70)

    try:
        schema = ensure_active_schema()

        train_dataset = RansomwareGraphDataset(
            data_dir=data_dir,
            split="train",
            cache_dir=cache_dir,
            schema=schema,
            validate_graphs=True,
        )
        val_dataset = RansomwareGraphDataset(
            data_dir=data_dir,
            split="val",
            cache_dir=cache_dir,
            schema=schema,
            validate_graphs=True,
        )
        test_dataset = RansomwareGraphDataset(
            data_dir=data_dir,
            split="test",
            cache_dir=cache_dir,
            schema=schema,
            validate_graphs=True,
        )

        for dataset in [train_dataset, val_dataset, test_dataset]:
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Validating {dataset.split.upper()} split")
            logger.info("=" * 50)

            stats = dataset.get_statistics()
            logger.info(f"Graphs: {stats['num_graphs']}")
            logger.info(f"Ransomware: {stats['num_ransomware']}")
            logger.info(f"Benign: {stats['num_benign']}")
            logger.info(f"Other malware: {stats['num_other_malware']}")

            for i in range(min(2, len(dataset))):
                data, label = dataset[i]

                logger.info(f"\n  Sample {i + 1}: {CLASS_NAMES[label]}")
                logger.info(f"    Node types: {data.node_types}")
                logger.info(f"    Edge types: {len(data.edge_types)} types")

                for node_type in data.node_types:
                    if hasattr(data[node_type], "x"):
                        logger.info(
                            f"      {node_type}: "
                            f"{data[node_type].x.shape[0]} nodes, "
                            f"{data[node_type].x.shape[1]}-dim"
                        )

                for edge_type in data.edge_types:
                    if "temporal" in edge_type[1]:
                        num_edges = data[edge_type].edge_index.shape[1]
                        logger.info(f"      Temporal edges: {num_edges}")
                        if hasattr(data[edge_type], "edge_attr"):
                            logger.info(
                                f"        Edge features: {data[edge_type].edge_attr.shape[1]}-dim"
                            )
                        break

        logger.info("\n" + "=" * 70)
        logger.info("Testing HeteroData Batching")
        logger.info("=" * 70)

        train_loader, _, _ = create_dataloaders(
            data_dir=data_dir,
            cache_dir=cache_dir,
            batch_size=4,
            num_workers=0,
            schema=schema,
        )

        batch, labels = next(iter(train_loader))
        logger.info("\nBatch info:")
        logger.info(f"  Batch size: {batch.num_graphs}")
        logger.info(f"  Type: {type(batch)}")
        logger.info(f"  Node types: {batch.node_types}")
        logger.info(f"  Labels: {labels.tolist()}")
        logger.info(f"  Class distribution: {[CLASS_NAMES[l] for l in labels.tolist()]}")

        for node_type in batch.node_types:
            if hasattr(batch[node_type], "x"):
                logger.info(f"  {node_type} nodes in batch: {batch[node_type].x.shape[0]}")

        logger.info("\n✅ HeteroData validation complete!")
        return True

    except Exception as e:
        logger.error(f"\n❌ Validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    validate_dataset()