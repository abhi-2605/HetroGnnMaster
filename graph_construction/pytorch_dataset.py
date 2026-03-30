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
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Set

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch_geometric.data import Batch, HeteroData

import config
from config import METADATA_PATH
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


# ============================================================================
# Samplers
# ============================================================================

class SizeAwareBatchSampler(Sampler):
    """
    Legacy sampler:
    Each batch = 1 large graph + (batch_size-1) small graphs.
    Uses file size on disk as proxy.
    Optionally accepts per-sample weights for weighted oversampling.
    """

    def __init__(self, dataset, batch_size, shuffle=True, weights=None):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.weights = np.array(weights) if weights is not None else None

        print("📏 Measuring graph sizes (fast, disk-only)...")
        sizes = np.array([f.stat().st_size for f in dataset.graph_files], dtype=np.float32)

        self.sizes = sizes
        self.indices = np.argsort(sizes)

        n_large = max(1, len(self.indices) // batch_size)
        self.small_i = self.indices[:-n_large].tolist()
        self.large_i = self.indices[-n_large:].tolist()

        large_pct = 100 // batch_size
        small_pct = 100 - large_pct
        print(f"   Smallest     : {sizes[self.indices[0]] / 1024:.1f} KB")
        print(f"   Largest      : {sizes[self.indices[-1]] / 1024 / 1024:.1f} MB")
        print(f"   Batch layout : 1 large + {batch_size - 1} small per batch")
        print(f"   Large bucket : {len(self.large_i)} graphs (top {large_pct}%)")
        print(f"   Small bucket : {len(self.small_i)} graphs (bot {small_pct}%)")
        print(f"   Batches      : {len(self):,}")

    def __iter__(self):
        small_i = self.small_i.copy()
        large_i = self.large_i.copy()
        n_small_per_batch = self.batch_size - 1
        n_batches = len(large_i)

        if self.weights is not None:
            small_w = self.weights[small_i]
            large_w = self.weights[large_i]
            small_w = small_w / small_w.sum()
            large_w = large_w / large_w.sum()

            small_pool = np.random.choice(
                small_i,
                size=n_batches * n_small_per_batch,
                p=small_w,
                replace=True,
            ).tolist()
            large_pool = np.random.choice(
                large_i,
                size=n_batches,
                p=large_w,
                replace=True,
            ).tolist()
        else:
            if self.shuffle:
                np.random.shuffle(small_i)
                np.random.shuffle(large_i)
            repeats = (n_batches * n_small_per_batch // max(1, len(small_i))) + 1
            small_pool = (small_i * repeats)[: n_batches * n_small_per_batch]
            large_pool = large_i[:n_batches]

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


class BucketedBatchSampler(Sampler):
    """
    Bucketed batch sampler based on API node counts.

    Uses per-graph API node count to assign samples into buckets, each with
    its own batch size.

    Expected config:
        ENABLE_BUCKETED_BATCH_SAMPLER = True
        BUCKETED_BATCH_RULES = [
            (10_000, 32),
            (25_000, 16),
            (50_000, 8),
            (100_000, 4),
            (10_000_000, 1),
        ]
    """

    def __init__(
        self,
        dataset,
        bucket_rules=None,
        shuffle=True,
        weights=None,
        verbose=True,
    ):
        self.dataset = dataset
        self.bucket_rules = bucket_rules or list(getattr(config, "BUCKETED_BATCH_RULES", []))
        self.shuffle = shuffle
        self.weights = np.array(weights, dtype=np.float64) if weights is not None else None
        self.verbose = verbose

        print("📏 Measuring API node counts for bucketed batching...")
        self.api_counts = self._measure_api_counts()
        self.buckets = self._build_buckets()

        self._num_batches = sum(
            int(np.ceil(len(bucket["indices"]) / bucket["batch_size"]))
            for bucket in self.buckets
            if len(bucket["indices"]) > 0
        )

        if self.verbose:
            self._print_summary()

    def _measure_api_counts(self) -> np.ndarray:
        counts = []

        for graph_path in self.dataset.graph_files:
            try:
                with open(graph_path, "rb") as f:
                    data = pickle.load(f)

                if isinstance(data, HeteroData) and "api" in data.node_types and hasattr(data["api"], "x"):
                    x = data["api"].x
                    n = int(x.shape[0]) if isinstance(x, torch.Tensor) else 0
                else:
                    n = 0
            except Exception:
                n = 0

            counts.append(n)

        return np.array(counts, dtype=np.int64)

    def _build_buckets(self):
        buckets = []
        remaining_indices = np.arange(len(self.api_counts))
        lower_bound = -1

        for max_nodes, batch_size in self.bucket_rules:
            mask = (
                (self.api_counts[remaining_indices] > lower_bound) &
                (self.api_counts[remaining_indices] <= max_nodes)
            )
            bucket_indices = remaining_indices[mask].tolist()

            buckets.append(
                {
                    "max_nodes": max_nodes,
                    "batch_size": int(batch_size),
                    "indices": bucket_indices,
                }
            )

            remaining_indices = remaining_indices[~mask]
            lower_bound = max_nodes

        if len(remaining_indices) > 0:
            fallback_bs = int(self.bucket_rules[-1][1]) if len(self.bucket_rules) > 0 else 1
            buckets.append(
                {
                    "max_nodes": float("inf"),
                    "batch_size": fallback_bs,
                    "indices": remaining_indices.tolist(),
                }
            )

        return buckets

    def _make_batches_for_bucket(self, bucket):
        indices = bucket["indices"].copy()
        batch_size = bucket["batch_size"]

        if len(indices) == 0:
            return []

        if self.weights is not None and getattr(config, "BUCKETED_BATCH_USE_WEIGHTS", True):
            bucket_weights = self.weights[indices]
            bucket_weights = bucket_weights / bucket_weights.sum()

            sampled = np.random.choice(
                indices,
                size=len(indices),
                replace=True,
                p=bucket_weights,
            ).tolist()

            if self.shuffle:
                np.random.shuffle(sampled)

            indices = sampled
        else:
            if self.shuffle:
                np.random.shuffle(indices)

        return [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

    def _print_summary(self):
        print("📦 Bucketed batch sampler summary:")
        total_graphs = len(self.api_counts)
        lower = 0

        for i, bucket in enumerate(self.buckets):
            idxs = bucket["indices"]
            bs = bucket["batch_size"]
            max_nodes = bucket["max_nodes"]

            count = len(idxs)
            avg_nodes = float(self.api_counts[idxs].mean()) if count > 0 else 0.0

            upper_txt = "inf" if max_nodes == float("inf") else f"{max_nodes:,}"
            if i == 0:
                label = f"<= {upper_txt}"
            else:
                label = f"{lower + 1:,} .. {upper_txt}"

            pct = 100.0 * count / max(1, total_graphs)
            print(
                f"   Bucket {i + 1}: api_nodes {label:<18} | "
                f"graphs={count:<5} ({pct:5.1f}%) | "
                f"batch_size={bs:<2} | avg_nodes={avg_nodes:,.1f}"
            )

            if max_nodes != float("inf"):
                lower = max_nodes

        print(f"   Total batches per epoch: {self._num_batches:,}")

    def __iter__(self):
        all_batches = []

        for bucket in self.buckets:
            all_batches.extend(self._make_batches_for_bucket(bucket))

        if self.shuffle:
            np.random.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self):
        return self._num_batches


CLASS_NAMES = {
    0: "Ransomware",
    1: "Benign",
    2: "Other Malware",
}


# ============================================================================
# Dataset
# ============================================================================

class RansomwareGraphDataset(Dataset):
    """
    Dataset for Temporal Heterogeneous Graphs.
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
        self.enable_augmentation = split == "train"
        self.validate_graphs = validate_graphs

        self.schema = schema if schema is not None else ensure_active_schema()
        self.canonical_node_dims = get_canonical_node_dims(self.schema)
        self.node_type_order = get_node_types(self.schema)

        self.graph_files = sorted(list(self.split_dir.glob("*.pkl")))
        if len(self.graph_files) == 0:
            raise ValueError(f"No .pkl files found in {self.split_dir}")

        logger.info(f"Loaded {len(self.graph_files)} HeteroData graphs for {split} split")
        logger.info(f"Using graph root: {self.cache_dir}")

        self.labels = self._load_labels()

    def _load_labels(self) -> Dict[str, int]:
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

    # ----------------------------------------------------------------------
    # Timestamp recovery
    # ----------------------------------------------------------------------

    def _ensure_api_timestamps(self, data: HeteroData) -> HeteroData:
        """
        Ensure data['api'].timestamps exists and matches API node count.
        Fallback: reconstruct from temporal api->api edge_attr[:, 0].
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

    # ----------------------------------------------------------------------
    # Auxiliary tensors
    # ----------------------------------------------------------------------

    def _ensure_api_aux_tensors(self, data: HeteroData) -> HeteroData:
        """
        Ensure repeat_count and run_duration exist on api store so PyG batching
        stays consistent across compressed/capped and untouched graphs.
        """
        if "api" not in data.node_types or not hasattr(data["api"], "x"):
            return data

        n = int(data["api"].x.shape[0])

        if not hasattr(data["api"], "repeat_count") or not isinstance(data["api"].repeat_count, torch.Tensor):
            data["api"].repeat_count = torch.ones(n, dtype=torch.float32)

        if not hasattr(data["api"], "run_duration") or not isinstance(data["api"].run_duration, torch.Tensor):
            data["api"].run_duration = torch.zeros(n, dtype=torch.float32)

        if data["api"].repeat_count.numel() != n:
            data["api"].repeat_count = torch.ones(n, dtype=torch.float32)

        if data["api"].run_duration.numel() != n:
            data["api"].run_duration = torch.zeros(n, dtype=torch.float32)

        return data

    # ----------------------------------------------------------------------
    # Conservative run compression
    # ----------------------------------------------------------------------

    def _build_api_local_signatures(self, data: HeteroData, num_api: int) -> List[tuple]:
        signatures = [defaultdict(int) for _ in range(num_api)]

        for edge_type in data.edge_types:
            src_type, rel, dst_type = edge_type
            store = data[edge_type]

            if not hasattr(store, "edge_index"):
                continue
            edge_index = store.edge_index
            if not isinstance(edge_index, torch.Tensor) or edge_index.numel() == 0:
                continue

            src_idx = edge_index[0].detach().cpu().tolist()
            dst_idx = edge_index[1].detach().cpu().tolist()

            if src_type == "api" and dst_type != "api":
                for s in src_idx:
                    if 0 <= s < num_api:
                        signatures[s][f"out::{rel}::{dst_type}"] += 1

            elif src_type != "api" and dst_type == "api":
                for d in dst_idx:
                    if 0 <= d < num_api:
                        signatures[d][f"in::{src_type}::{rel}"] += 1

            elif src_type == "api" and dst_type == "api" and rel != "temporal":
                for s, d in zip(src_idx, dst_idx):
                    if 0 <= s < num_api:
                        signatures[s][f"out::{rel}::api"] += 1
                    if 0 <= d < num_api:
                        signatures[d][f"in::api::{rel}"] += 1

        return [tuple(sorted(sig.items())) for sig in signatures]

    def _api_nodes_are_compressible(
        self,
        data: HeteroData,
        i: int,
        j: int,
        local_signatures: List[tuple],
    ) -> bool:
        if "api" not in data.node_types or not hasattr(data["api"], "x"):
            return False

        api_store = data["api"]
        x = api_store.x
        if not isinstance(x, torch.Tensor):
            return False

        if getattr(config, "API_RUN_COMPRESSION_REQUIRE_API_NAMES", True):
            if not hasattr(api_store, "api_names"):
                return False
            try:
                if api_store.api_names[i] != api_store.api_names[j]:
                    return False
            except Exception:
                return False

        meta_start = int(getattr(config, "API_RUN_COMPRESSION_METADATA_START_DIM", 768))
        atol = float(getattr(config, "API_RUN_COMPRESSION_METADATA_ATOL", 1e-6))
        if x.size(1) > meta_start:
            if not torch.allclose(x[i, meta_start:], x[j, meta_start:], atol=atol, rtol=0.0):
                return False

        if local_signatures[i] != local_signatures[j]:
            return False

        if getattr(config, "API_RUN_COMPRESSION_USE_TIME_GAP", True):
            if hasattr(api_store, "timestamps") and isinstance(api_store.timestamps, torch.Tensor):
                ts = api_store.timestamps
                if ts.numel() > max(i, j):
                    max_gap = float(getattr(config, "API_RUN_COMPRESSION_MAX_TIME_GAP", 1.0))
                    gap = float(ts[j].item() - ts[i].item())
                    if gap < 0 or gap > max_gap:
                        return False

        return True

    def _rebuild_temporal_chain_after_compression(
        self,
        data: HeteroData,
        runs: List[Tuple[int, int]],
    ) -> HeteroData:
        temporal_edge_type = ("api", "temporal", "api")
        if temporal_edge_type not in data.edge_types:
            return data

        api_store = data["api"]
        temporal_store = data[temporal_edge_type]

        edge_attr_dim = 0
        edge_attr_dtype = torch.float32
        edge_attr_device = api_store.x.device

        old_temporal_attr_map = {}
        if hasattr(temporal_store, "edge_index") and isinstance(temporal_store.edge_index, torch.Tensor):
            old_ei = temporal_store.edge_index.detach().cpu()
            if hasattr(temporal_store, "edge_attr") and isinstance(temporal_store.edge_attr, torch.Tensor):
                old_ea = temporal_store.edge_attr
                edge_attr_dim = old_ea.size(1) if old_ea.dim() == 2 else 0
                edge_attr_dtype = old_ea.dtype
                edge_attr_device = old_ea.device

                for k in range(old_ei.size(1)):
                    s = int(old_ei[0, k].item())
                    d = int(old_ei[1, k].item())
                    old_temporal_attr_map[(s, d)] = old_ea[k]

        new_num_api = len(runs)
        if new_num_api <= 1:
            temporal_store.edge_index = torch.empty((2, 0), dtype=torch.long)
            if edge_attr_dim > 0:
                temporal_store.edge_attr = torch.empty(
                    (0, edge_attr_dim),
                    dtype=edge_attr_dtype,
                    device=edge_attr_device,
                )
            return data

        new_src = []
        new_dst = []
        new_edge_attrs = []

        for new_idx in range(new_num_api - 1):
            cur_start, cur_end = runs[new_idx]
            nxt_start, _ = runs[new_idx + 1]

            new_src.append(new_idx)
            new_dst.append(new_idx + 1)

            if edge_attr_dim > 0:
                boundary_attr = old_temporal_attr_map.get((cur_end, nxt_start), None)
                if boundary_attr is None:
                    boundary_attr = torch.zeros(edge_attr_dim, dtype=edge_attr_dtype, device=edge_attr_device)
                new_edge_attrs.append(boundary_attr)

        temporal_store.edge_index = torch.tensor([new_src, new_dst], dtype=torch.long)
        if edge_attr_dim > 0:
            temporal_store.edge_attr = torch.stack(new_edge_attrs, dim=0)

        return data

    def _compress_consecutive_api_runs(self, data: HeteroData) -> HeteroData:
        if not getattr(config, "ENABLE_API_RUN_COMPRESSION", False):
            return self._ensure_api_aux_tensors(data)

        if "api" not in data.node_types:
            return data
        if not hasattr(data["api"], "x") or not isinstance(data["api"].x, torch.Tensor):
            return data

        num_api = int(data["api"].x.shape[0])
        if num_api <= 1:
            return self._ensure_api_aux_tensors(data)

        if getattr(config, "API_RUN_COMPRESSION_REQUIRE_API_NAMES", True):
            if not hasattr(data["api"], "api_names"):
                return self._ensure_api_aux_tensors(data)
            try:
                if len(data["api"].api_names) != num_api:
                    return self._ensure_api_aux_tensors(data)
            except Exception:
                return self._ensure_api_aux_tensors(data)

        local_signatures = self._build_api_local_signatures(data, num_api)

        runs: List[Tuple[int, int]] = []
        run_start = 0
        for idx in range(1, num_api):
            if self._api_nodes_are_compressible(data, idx - 1, idx, local_signatures):
                continue
            runs.append((run_start, idx - 1))
            run_start = idx
        runs.append((run_start, num_api - 1))

        if len(runs) == num_api:
            return self._ensure_api_aux_tensors(data)

        api_store = data["api"]
        old_to_new = torch.empty(num_api, dtype=torch.long)
        keep_indices = []
        repeat_counts = []
        run_durations = []

        for new_idx, (start, end) in enumerate(runs):
            keep_indices.append(start)
            repeat_counts.append(end - start + 1)

            for old_idx in range(start, end + 1):
                old_to_new[old_idx] = new_idx

            if hasattr(api_store, "timestamps") and isinstance(api_store.timestamps, torch.Tensor):
                duration = float(api_store.timestamps[end].item() - api_store.timestamps[start].item())
            else:
                duration = 0.0
            run_durations.append(duration)

        keep_indices_t = torch.tensor(keep_indices, dtype=torch.long)

        api_store.x = api_store.x[keep_indices_t]

        if hasattr(api_store, "timestamps") and isinstance(api_store.timestamps, torch.Tensor):
            api_store.timestamps = api_store.timestamps[keep_indices_t]

        if hasattr(api_store, "api_names"):
            try:
                api_store.api_names = [api_store.api_names[i] for i in keep_indices]
            except Exception:
                pass

        api_store.repeat_count = torch.tensor(repeat_counts, dtype=torch.float32)
        api_store.run_duration = torch.tensor(run_durations, dtype=torch.float32)
        api_store.num_nodes = len(runs)

        temporal_edge_type = ("api", "temporal", "api")
        for edge_type in list(data.edge_types):
            if edge_type == temporal_edge_type:
                continue

            src_type, _, dst_type = edge_type
            store = data[edge_type]

            if not hasattr(store, "edge_index") or not isinstance(store.edge_index, torch.Tensor):
                continue
            if store.edge_index.numel() == 0:
                continue

            edge_index = store.edge_index.clone()
            if src_type == "api":
                edge_index[0] = old_to_new[edge_index[0]]
            if dst_type == "api":
                edge_index[1] = old_to_new[edge_index[1]]
            store.edge_index = edge_index

            if hasattr(store, "edge_attr") and isinstance(store.edge_attr, torch.Tensor):
                if store.edge_attr.size(0) == store.edge_index.size(1):
                    pass

        data = self._rebuild_temporal_chain_after_compression(data, runs)
        return self._ensure_api_aux_tensors(data)

    # ----------------------------------------------------------------------
    # Behavior-aware cap
    # ----------------------------------------------------------------------

    def _get_behavior_protected_api_indices(self, data: HeteroData, num_api: int) -> Set[int]:
        protected: Set[int] = set()
        if num_api <= 0:
            return protected

        protected.add(0)
        protected.add(num_api - 1)

        api_store = data["api"]

        if hasattr(api_store, "api_names"):
            try:
                api_names = api_store.api_names
                for i in range(1, min(len(api_names), num_api)):
                    if api_names[i] != api_names[i - 1]:
                        protected.add(i - 1)
                        protected.add(i)
            except Exception:
                pass

        if (
            getattr(config, "BEHAVIOR_AWARE_API_CAP_PROTECT_TIME_SPIKES", True)
            and hasattr(api_store, "timestamps")
            and isinstance(api_store.timestamps, torch.Tensor)
        ):
            ts = api_store.timestamps
            if ts.numel() == num_api and num_api > 2:
                diffs = ts[1:] - ts[:-1]
                try:
                    if diffs.numel() > 0:
                        qv = float(getattr(config, "BEHAVIOR_AWARE_API_CAP_TIME_SPIKE_QUANTILE", 0.95))
                        q = torch.quantile(diffs.float(), qv)
                        spike_idx = (diffs >= q).nonzero(as_tuple=False).view(-1).tolist()
                        for idx in spike_idx:
                            protected.add(int(idx))
                            protected.add(int(idx + 1))
                except Exception:
                    pass

        if getattr(config, "BEHAVIOR_AWARE_API_CAP_PROTECT_EDGE_ENDPOINTS", True):
            for edge_type in data.edge_types:
                src_type, rel, dst_type = edge_type
                if rel == "temporal":
                    continue

                store = data[edge_type]
                if not hasattr(store, "edge_index") or not isinstance(store.edge_index, torch.Tensor):
                    continue
                if store.edge_index.numel() == 0:
                    continue

                ei = store.edge_index
                if src_type == "api":
                    protected.update(int(v) for v in ei[0].detach().cpu().tolist() if 0 <= int(v) < num_api)
                if dst_type == "api":
                    protected.update(int(v) for v in ei[1].detach().cpu().tolist() if 0 <= int(v) < num_api)

        return protected

    def _build_segment_ranges(self, keep_indices: List[int], n: int) -> List[Tuple[int, int]]:
        """
        Each kept node represents a local interval around itself.
        Used to aggregate repeat_count and run_duration after behavior-aware cap.
        """
        if not keep_indices:
            return []

        segments: List[Tuple[int, int]] = []
        for i, cur in enumerate(keep_indices):
            if i == 0:
                start = 0
            else:
                prev = keep_indices[i - 1]
                start = (prev + cur) // 2 + 1

            if i == len(keep_indices) - 1:
                end = n - 1
            else:
                nxt = keep_indices[i + 1]
                end = (cur + nxt) // 2

            start = max(0, min(start, n - 1))
            end = max(start, min(end, n - 1))
            segments.append((start, end))

        return segments

    def _apply_api_subset(
        self,
        data: HeteroData,
        keep_indices: List[int],
        aggregate_segments: Optional[List[Tuple[int, int]]] = None,
    ) -> HeteroData:
        """
        Apply an ordered subset of API nodes and remap all API-linked edges.
        If aggregate_segments is provided, aggregate repeat_count/run_duration
        over represented regions instead of simple indexing.
        """
        if "api" not in data.node_types or not hasattr(data["api"], "x"):
            return data

        keep_indices = sorted(set(int(i) for i in keep_indices))
        api_store = data["api"]
        old_num_api = int(api_store.x.shape[0])

        if old_num_api == 0 or len(keep_indices) == 0 or len(keep_indices) >= old_num_api:
            return self._ensure_api_aux_tensors(data)

        keep_t = torch.tensor(keep_indices, dtype=torch.long)
        old_to_new = torch.full((old_num_api,), -1, dtype=torch.long)
        old_to_new[keep_t] = torch.arange(len(keep_indices), dtype=torch.long)

        old_x = api_store.x
        old_ts = (
            api_store.timestamps.clone()
            if hasattr(api_store, "timestamps") and isinstance(api_store.timestamps, torch.Tensor)
            else None
        )
        old_repeat = (
            api_store.repeat_count.clone()
            if hasattr(api_store, "repeat_count") and isinstance(api_store.repeat_count, torch.Tensor)
            else torch.ones(old_num_api, dtype=torch.float32)
        )
        old_run_dur = (
            api_store.run_duration.clone()
            if hasattr(api_store, "run_duration") and isinstance(api_store.run_duration, torch.Tensor)
            else torch.zeros(old_num_api, dtype=torch.float32)
        )
        old_api_names = list(api_store.api_names) if hasattr(api_store, "api_names") else None

        api_store.x = old_x[keep_t]

        if old_ts is not None:
            api_store.timestamps = old_ts[keep_t]

        if old_api_names is not None:
            try:
                api_store.api_names = [old_api_names[i] for i in keep_indices]
            except Exception:
                pass

        if aggregate_segments is None:
            api_store.repeat_count = old_repeat[keep_t].float()
            api_store.run_duration = old_run_dur[keep_t].float()
        else:
            agg_repeat = []
            agg_duration = []

            for start, end in aggregate_segments:
                agg_repeat.append(float(old_repeat[start:end + 1].sum().item()))
                if old_ts is not None:
                    agg_duration.append(float(old_ts[end].item() - old_ts[start].item()))
                else:
                    agg_duration.append(float(old_run_dur[start:end + 1].sum().item()))

            api_store.repeat_count = torch.tensor(agg_repeat, dtype=torch.float32)
            api_store.run_duration = torch.tensor(agg_duration, dtype=torch.float32)

        api_store.num_nodes = len(keep_indices)

        temporal_edge_type = ("api", "temporal", "api")
        old_temporal_attr_map = {}
        edge_attr_dim = 0
        edge_attr_dtype = torch.float32
        edge_attr_device = api_store.x.device

        if temporal_edge_type in data.edge_types:
            old_temporal_store = data[temporal_edge_type]
            if hasattr(old_temporal_store, "edge_index") and isinstance(old_temporal_store.edge_index, torch.Tensor):
                old_ei = old_temporal_store.edge_index.detach().cpu()
                if hasattr(old_temporal_store, "edge_attr") and isinstance(old_temporal_store.edge_attr, torch.Tensor):
                    old_ea = old_temporal_store.edge_attr
                    if old_ea.dim() == 2:
                        edge_attr_dim = old_ea.size(1)
                    edge_attr_dtype = old_ea.dtype
                    edge_attr_device = old_ea.device
                    for k in range(old_ei.size(1)):
                        s = int(old_ei[0, k].item())
                        d = int(old_ei[1, k].item())
                        old_temporal_attr_map[(s, d)] = old_ea[k]

        for edge_type in list(data.edge_types):
            src_type, _, dst_type = edge_type
            store = data[edge_type]

            if not hasattr(store, "edge_index") or not isinstance(store.edge_index, torch.Tensor):
                continue

            ei = store.edge_index
            if ei.numel() == 0:
                continue

            mask = torch.ones(ei.size(1), dtype=torch.bool)

            if src_type == "api":
                mask &= old_to_new[ei[0]] >= 0
            if dst_type == "api":
                mask &= old_to_new[ei[1]] >= 0

            new_ei = ei[:, mask].clone()

            if src_type == "api":
                new_ei[0] = old_to_new[new_ei[0]]
            if dst_type == "api":
                new_ei[1] = old_to_new[new_ei[1]]

            store.edge_index = new_ei

            if hasattr(store, "edge_attr") and isinstance(store.edge_attr, torch.Tensor):
                if store.edge_attr.size(0) == ei.size(1):
                    store.edge_attr = store.edge_attr[mask]

        if temporal_edge_type in data.edge_types:
            temporal_store = data[temporal_edge_type]
            new_num_api = len(keep_indices)

            if new_num_api <= 1:
                temporal_store.edge_index = torch.empty((2, 0), dtype=torch.long)
                if edge_attr_dim > 0:
                    temporal_store.edge_attr = torch.empty(
                        (0, edge_attr_dim),
                        dtype=edge_attr_dtype,
                        device=edge_attr_device,
                    )
            else:
                new_src = []
                new_dst = []
                new_edge_attrs = []

                for new_idx in range(new_num_api - 1):
                    old_s = keep_indices[new_idx]
                    old_d = keep_indices[new_idx + 1]
                    new_src.append(new_idx)
                    new_dst.append(new_idx + 1)

                    if edge_attr_dim > 0:
                        boundary_attr = old_temporal_attr_map.get((old_s, old_d), None)
                        if boundary_attr is None:
                            boundary_attr = torch.zeros(edge_attr_dim, dtype=edge_attr_dtype, device=edge_attr_device)
                            if old_ts is not None and old_ts.numel() > max(old_s, old_d):
                                boundary_attr = boundary_attr.clone()
                                boundary_attr[0] = old_ts[old_d] - old_ts[old_s]
                        new_edge_attrs.append(boundary_attr)

                temporal_store.edge_index = torch.tensor([new_src, new_dst], dtype=torch.long)
                if edge_attr_dim > 0:
                    temporal_store.edge_attr = torch.stack(new_edge_attrs, dim=0)

        return self._ensure_api_aux_tensors(data)

    def _behavior_aware_api_cap(self, data: HeteroData) -> HeteroData:
        """
        Cap graphs only when API count exceeds threshold, while preserving
        behavior landmarks and temporal structure.
        """
        if not getattr(config, "ENABLE_BEHAVIOR_AWARE_API_CAP", False):
            return self._ensure_api_aux_tensors(data)

        if "api" not in data.node_types or not hasattr(data["api"], "x"):
            return data

        num_api = int(data["api"].x.shape[0])
        threshold = int(getattr(config, "BEHAVIOR_AWARE_API_CAP_THRESHOLD", 100000))
        max_nodes = int(getattr(config, "BEHAVIOR_AWARE_API_CAP_MAX_NODES", 25000))

        if num_api <= threshold or num_api <= max_nodes:
            return self._ensure_api_aux_tensors(data)

        if getattr(config, "BEHAVIOR_AWARE_API_CAP_REQUIRE_API_NAMES", False):
            if not hasattr(data["api"], "api_names"):
                return self._ensure_api_aux_tensors(data)

        protected = self._get_behavior_protected_api_indices(data, num_api)
        protected_sorted = sorted(i for i in protected if 0 <= i < num_api)

        if len(protected_sorted) >= max_nodes:
            pos = np.linspace(0, len(protected_sorted) - 1, num=max_nodes, dtype=int).tolist()
            keep_indices = sorted({protected_sorted[p] for p in pos})
        else:
            keep_set = set(protected_sorted)
            remaining_budget = max_nodes - len(keep_set)

            candidates = [i for i in range(num_api) if i not in keep_set]
            if remaining_budget > 0 and len(candidates) > 0:
                if remaining_budget >= len(candidates):
                    sampled = candidates
                else:
                    pos = np.linspace(0, len(candidates) - 1, num=remaining_budget, dtype=int).tolist()
                    sampled = [candidates[p] for p in pos]
                keep_set.update(sampled)

            keep_indices = sorted(keep_set)

        if 0 not in keep_indices:
            keep_indices = [0] + keep_indices
        if (num_api - 1) not in keep_indices:
            keep_indices.append(num_api - 1)

        if len(keep_indices) > max_nodes:
            pos = np.linspace(0, len(keep_indices) - 1, num=max_nodes, dtype=int).tolist()
            keep_indices = sorted({keep_indices[p] for p in pos})
            if 0 not in keep_indices:
                keep_indices[0] = 0
            if (num_api - 1) not in keep_indices:
                keep_indices[-1] = num_api - 1

        segments = self._build_segment_ranges(keep_indices, num_api)

        logger.info(
            f"🔧 Behavior-aware API cap: {num_api} -> {len(keep_indices)} "
            f"(protected={len(protected_sorted)}, threshold={threshold}, max={max_nodes})"
        )

        return self._apply_api_subset(data, keep_indices, aggregate_segments=segments)

    # ----------------------------------------------------------------------
    # Public dataset path
    # ----------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Tuple[HeteroData, int]:
        graph_path = self.graph_files[idx]

        with open(graph_path, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, HeteroData):
            raise TypeError(f"Expected HeteroData, got {type(data)}")

        hash_id = graph_path.stem

        data = self._ensure_api_timestamps(data)

        label = self.labels.get(hash_id, 2)
        if hash_id not in self.labels:
            logger.warning(f"No label found for {hash_id}, defaulting to 2")

        if config.NORMALIZE_NODE_FEATURE_DIMS:
            data = normalize_graph_node_features(data, schema=self.schema, inplace=True)

        data = self._compress_consecutive_api_runs(data)
        data = self._behavior_aware_api_cap(data)
        data = self._ensure_api_aux_tensors(data)

        if self.validate_graphs and config.FAIL_ON_SCHEMA_MISMATCH:
            validate_graph_or_raise(
                data,
                schema=self.schema,
                require_api_nodes=True,
                strict=config.SCHEMA_STRICT_VALIDATION,
            )

        if self.enable_augmentation and label == 1:
            if random.random() < 0.7:
                data = self.augment_benign_graph(data)
                if config.NORMALIZE_NODE_FEATURE_DIMS:
                    data = normalize_graph_node_features(data, schema=self.schema, inplace=True)
                data = self._ensure_api_aux_tensors(data)

        return data, label

    def augment_benign_graph(self, data: HeteroData) -> HeteroData:
        data = copy.deepcopy(data)

        resource_node_types = ["file", "registry", "process", "network"]

        for nt in resource_node_types:
            if nt in data.node_types and hasattr(data[nt], "x"):
                x = data[nt].x
                if isinstance(x, torch.Tensor) and x.numel() > 0:
                    noise = 0.01 * torch.randn_like(x)
                    x = x + noise
                    mask = (torch.rand_like(x) > 0.05).float()
                    x = x * mask
                    data[nt].x = x

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


# ============================================================================
# Collate helpers
# ============================================================================

def _clean_graph_for_batching(graph: HeteroData) -> HeteroData:
    """
    Remove non-tensor attrs that break PyG batching, while preserving
    tensors needed later.
    """
    keep_graph_level = set()
    keep_store_attrs = {
        "x",
        "timestamps",
        "timestamp",
        "time",
        "num_nodes",
        "repeat_count",
        "run_duration",
    }
    drop_store_attrs = {
        "placeholder_count",
        "num_apis",
        "graph_features",
    }

    top_keys_to_remove = []
    for key in list(graph.keys()):
        if key not in keep_graph_level:
            top_keys_to_remove.append(key)

    for key in top_keys_to_remove:
        try:
            del graph[key]
        except Exception:
            pass

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

    for node_type in graph.node_types:
        if hasattr(graph[node_type], "x") and isinstance(graph[node_type].x, torch.Tensor):
            graph[node_type].num_nodes = graph[node_type].x.shape[0]

    return graph


def hetero_collate(
    batch,
    schema: Optional[Dict] = None,
    validate_batch: bool = True,
):
    schema = schema if schema is not None else get_active_schema()

    if not batch:
        return None, None

    graphs = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)

    all_api_names = []
    for graph in graphs:
        if "api" in graph.node_types and hasattr(graph["api"], "api_names"):
            try:
                all_api_names.extend(graph["api"].api_names)
            except Exception:
                pass

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
        if graph is None:
            logger.warning("_clean_graph_for_batching returned None — skipping graph")
            continue

        processed_graphs.append(graph)

    if not processed_graphs:
        logger.warning("hetero_collate: all graphs failed cleaning — skipping batch")
        return None, None

    batched_graph = Batch.from_data_list(processed_graphs)

    if all_api_names:
        batched_graph["api"].api_names = all_api_names

    batched_graph.graph_timestamps = timestamps_dict if timestamps_dict else None

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


# ============================================================================
# Dataloaders
# ============================================================================

def create_dataloaders(
    data_dir: str = "data",
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle_train: bool = True,
    schema: Optional[Dict] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
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

        collate = lambda batch: hetero_collate(batch, schema=schema, validate_batch=True)

        sample_weights = None
        if getattr(config, "USE_WEIGHTED_SAMPLING", False):
            sample_weights = [
                config.OVERSAMPLE_WEIGHTS.get(train_dataset.labels.get(graph_file.stem, 2), 1.0)
                for graph_file in train_dataset.graph_files
            ]

        if getattr(config, "ENABLE_BUCKETED_BATCH_SAMPLER", False):
            train_batch_sampler = BucketedBatchSampler(
                train_dataset,
                bucket_rules=getattr(config, "BUCKETED_BATCH_RULES", None),
                shuffle=getattr(config, "BUCKETED_BATCH_SHUFFLE", True),
                weights=sample_weights,
                verbose=getattr(config, "BUCKETED_BATCH_VERBOSE", True),
            )

            train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_batch_sampler,
                num_workers=num_workers,
                collate_fn=collate,
                pin_memory=getattr(config, "PIN_MEMORY", False),
                prefetch_factor=getattr(config, "PREFETCH_FACTOR", 2) if num_workers > 0 else None,
                persistent_workers=getattr(config, "PERSISTENT_WORKERS", False) if num_workers > 0 else False,
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle_train,
                num_workers=num_workers,
                collate_fn=collate,
                pin_memory=getattr(config, "PIN_MEMORY", False),
                prefetch_factor=getattr(config, "PREFETCH_FACTOR", 2) if num_workers > 0 else None,
                persistent_workers=getattr(config, "PERSISTENT_WORKERS", False) if num_workers > 0 else False,
            )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=getattr(config, "PIN_MEMORY", False),
            prefetch_factor=getattr(config, "PREFETCH_FACTOR", 2) if num_workers > 0 else None,
            persistent_workers=getattr(config, "PERSISTENT_WORKERS", False) if num_workers > 0 else False,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate,
            pin_memory=getattr(config, "PIN_MEMORY", False),
            prefetch_factor=getattr(config, "PREFETCH_FACTOR", 2) if num_workers > 0 else None,
            persistent_workers=getattr(config, "PERSISTENT_WORKERS", False) if num_workers > 0 else False,
        )

        return train_loader, val_loader, test_loader

    except Exception as e:
        logger.error(f"Error creating dataloaders: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


# ============================================================================
# Validation utility
# ============================================================================

def validate_dataset(data_dir: str = "data", cache_dir: Optional[str] = None):
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

                if "api" in data.node_types and hasattr(data["api"], "repeat_count"):
                    logger.info(f"      api.repeat_count shape: {tuple(data['api'].repeat_count.shape)}")
                if "api" in data.node_types and hasattr(data["api"], "run_duration"):
                    logger.info(f"      api.run_duration shape: {tuple(data['api'].run_duration.shape)}")

                for edge_type in data.edge_types:
                    if "temporal" in edge_type[1]:
                        num_edges = data[edge_type].edge_index.shape[1]
                        logger.info(f"      Temporal edges: {num_edges}")
                        if hasattr(data[edge_type], "edge_attr"):
                            logger.info(f"        Edge features: {data[edge_type].edge_attr.shape[1]}-dim")
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
        if batch is None or labels is None:
            raise RuntimeError("First batch is None after collate")

        logger.info("\nBatch info:")
        logger.info(f"  Batch size: {batch.num_graphs}")
        logger.info(f"  Type: {type(batch)}")
        logger.info(f"  Node types: {batch.node_types}")
        logger.info(f"  Labels: {labels.tolist()}")
        logger.info(f"  Class distribution: {[CLASS_NAMES[l] for l in labels.tolist()]}")

        for node_type in batch.node_types:
            if hasattr(batch[node_type], "x"):
                logger.info(f"  {node_type} nodes in batch: {batch[node_type].x.shape[0]}")

        if "api" in batch.node_types and hasattr(batch["api"], "repeat_count"):
            logger.info(f"  api.repeat_count in batch: {tuple(batch['api'].repeat_count.shape)}")
        if "api" in batch.node_types and hasattr(batch["api"], "run_duration"):
            logger.info(f"  api.run_duration in batch: {tuple(batch['api'].run_duration.shape)}")

        logger.info("\n✅ HeteroData validation complete!")
        return True

    except Exception as e:
        logger.error(f"\n❌ Validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    validate_dataset()