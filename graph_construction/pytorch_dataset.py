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
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Batch, HeteroData

from config import METADATA_PATH
import numpy as np
from torch.utils.data import Sampler
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
        """Load 3-class labels from metadata CSV using md5 + all_categories."""
        metadata_path = Path(METADATA_PATH)

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        df = pd.read_csv(metadata_path)

        if "md5" not in df.columns:
            raise ValueError("Metadata CSV must contain column 'md5'")
        if "all_categories" not in df.columns:
            raise ValueError("Metadata CSV must contain column 'all_categories'")

        labels: Dict[str, int] = {}
        for _, row in df.iterrows():
            hash_id = str(row["md5"]).strip()
            category_names = str(row["all_categories"]).lower()

            if "ransomware" in category_names:
                label = 0
            elif "benign" in category_names:
                label = 1
            else:
                label = 2

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
        Ensure data['api'].timestamps exists and matches api node count.

        Fallback:
        reconstruct relative timestamps from temporal api->api edge_attr[:, 0]
        only when explicit api.timestamps is missing or misaligned.
        """
        if "api" not in data.node_types:
            return data

        if hasattr(data["api"], "timestamps"):
            ts = data["api"].timestamps
            if ts is not None and hasattr(data["api"], "x"):
                if isinstance(ts, torch.Tensor) and ts.numel() == data["api"].x.shape[0]:
                    return data

        edge_type = ("api", "temporal", "api")
        if edge_type not in data.edge_types:
            return data

        store = data[edge_type]

        if not hasattr(store, "edge_index") or store.edge_index is None:
            return data
        if not hasattr(store, "edge_attr") or store.edge_attr is None:
            return data
        if not hasattr(data["api"], "x") or data["api"].x is None:
            return data

        edge_index = store.edge_index
        edge_attr = store.edge_attr

        if not isinstance(edge_index, torch.Tensor) or edge_index.dim() != 2 or edge_index.size(0) != 2:
            return data
        if not isinstance(edge_attr, torch.Tensor) or edge_attr.dim() != 2 or edge_attr.size(1) < 1:
            return data

        num_api = int(data["api"].x.shape[0])
        if num_api == 0:
            data["api"].timestamps = torch.empty(0, dtype=torch.float32)
            return data

        src = edge_index[0].detach().cpu().tolist()
        dst = edge_index[1].detach().cpu().tolist()
        deltas = edge_attr[:, 0].detach().cpu().float().tolist()

        incoming = {}
        outgoing = {}
        for s, d, dt in zip(src, dst, deltas):
            outgoing.setdefault(s, []).append((d, dt))
            if d not in incoming:
                incoming[d] = (s, dt)

        start_nodes = [i for i in range(num_api) if i not in incoming]
        if not start_nodes:
            start_nodes = [0]

        timestamps = [None] * num_api
        for s in start_nodes:
            timestamps[s] = 0.0

        stack = list(start_nodes)
        while stack:
            u = stack.pop()
            base_t = timestamps[u]
            for v, dt in outgoing.get(u, []):
                cand = float(base_t) + float(dt)
                if timestamps[v] is None or cand < timestamps[v]:
                    timestamps[v] = cand
                    stack.append(v)

        last_t = 0.0
        for i in range(num_api):
            if timestamps[i] is None:
                timestamps[i] = last_t
            last_t = timestamps[i]

        data["api"].timestamps = torch.tensor(timestamps, dtype=torch.float32)

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

        # Restore api.timestamps when missing
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
                sample = self._ensure_api_timestamps(sample)

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
    """
    Remove non-tensor attrs that break PyG batching, while preserving attrs
    needed later by train.py / tgnn_model.py.
    """
    keep_graph_level = set()
    keep_store_attrs = {"timestamps", "timestamp", "time", "num_nodes"}
    drop_store_attrs = {"placeholder_count", "num_apis", "graph_features"}

    # Clean graph-level attrs
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

            # Keep tensor attrs that are explicitly needed
            if key in keep_store_attrs:
                continue

            # Keep x
            if key == "x":
                continue

            # Keep api_names only at graph level before batching harvest; it will be removed here
            # because Batch.from_data_list cannot safely collate list[str].
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
            if key in drop_store_attrs:
                keys_to_remove.append(key)
                continue

            if key in {"edge_index", "edge_attr"}:
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

    # Ensure num_nodes is explicitly set
    for node_type in graph.node_types:
        if hasattr(graph[node_type], "x") and isinstance(graph[node_type].x, torch.Tensor):
            graph[node_type].num_nodes = graph[node_type].x.shape[0]

    return graph


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

    # Preserve API names before cleaning
    all_api_names = []
    for graph in graphs:
        if "api" in graph.node_types and hasattr(graph["api"], "api_names"):
            try:
                all_api_names.extend(graph["api"].api_names)
            except Exception:
                pass

    # Collect timestamps before cleaning
    timestamps_dict = {}
    if len(graphs) > 0:
        all_node_types = set()
        for graph in graphs:
            all_node_types.update(graph.node_types)

        for node_type in all_node_types:
            all_timestamps = []
            missing_for_this_type = False

            for graph in graphs:
                if node_type in graph.node_types:
                    if hasattr(graph[node_type], "timestamps"):
                        ts = graph[node_type].timestamps
                        if isinstance(ts, torch.Tensor):
                            all_timestamps.append(ts)
                    else:
                        missing_for_this_type = True

            # Only concatenate if every graph that has this node type also has timestamps
            # Otherwise Batch.from_data_list may fail or downstream alignment may break.
            if all_timestamps and not missing_for_this_type:
                try:
                    timestamps_dict[node_type] = torch.cat(all_timestamps)
                except Exception:
                    pass

    processed_graphs = []
    for graph in graphs:
        if config.NORMALIZE_NODE_FEATURE_DIMS:
            graph = normalize_graph_node_features(graph, schema=schema, inplace=True)

        if validate_batch and config.FAIL_ON_SCHEMA_MISMATCH:
            validate_graph_or_raise(
                graph,
                schema=schema,
                require_api_nodes=True,
                strict=config.SCHEMA_STRICT_VALIDATION,
            )

        graph = _clean_graph_for_batching(graph)
        processed_graphs.append(graph)

    batched_graph = Batch.from_data_list(processed_graphs)

    # Restore auxiliary attrs needed later
    if all_api_names:
        batched_graph["api"].api_names = all_api_names

    batched_graph.graph_timestamps = timestamps_dict if timestamps_dict else None

    # Build model-facing dictionaries
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
            require_api_indices_if_memory=False,
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