from __future__ import annotations

import traceback
from pathlib import Path

import torch

import config
from graph_construction.pytorch_dataset import (
    RansomwareGraphDataset,
    create_dataloaders,
    _clean_graph_for_batching,
)
from utils.schema_manager import ensure_active_schema


SUSPICIOUS_GRAPH_LEVEL_KEYS = {
    "num_apis",
    "graph_id",
    "label_name",
    "regime",
    "source_format",
    "target_sha256",
    "analysis_duration",
    "activity_features",
    "time_fallback_modes",
}


def graph_top_level_keys(graph) -> list[str]:
    try:
        return sorted(list(graph.keys()))
    except Exception:
        return []


def print_suspicious_keys(graph, prefix: str) -> None:
    keys = set(graph_top_level_keys(graph))
    found = sorted(keys & SUSPICIOUS_GRAPH_LEVEL_KEYS)
    if found:
        print(f"{prefix} suspicious graph-level keys: {found}")
    else:
        print(f"{prefix} suspicious graph-level keys: none")


def assert_clean_for_batching(graph) -> None:
    keys = set(graph_top_level_keys(graph))
    bad = sorted(keys & SUSPICIOUS_GRAPH_LEVEL_KEYS)
    assert not bad, f"Graph still has unsafe graph-level keys after cleaning: {bad}"


def validate_single_graph_structure(graph) -> None:
    assert "api" in graph.node_types, "Missing required node type 'api'"
    assert hasattr(graph["api"], "x"), "api.x missing"
    assert graph["api"].x.ndim == 2, "api.x must be 2D"
    assert graph["api"].x.shape[1] == config.API_NODE_DIM, (
        f"api.x dim mismatch: got {graph['api'].x.shape[1]}, "
        f"expected {config.API_NODE_DIM}"
    )

    temporal_edge = ("api", "temporal", "api")
    assert temporal_edge in graph.edge_types, "Missing temporal edge type"
    assert hasattr(graph[temporal_edge], "edge_index"), "Temporal edge_index missing"

    if hasattr(graph[temporal_edge], "edge_attr"):
        assert graph[temporal_edge].edge_attr.ndim == 2, "Temporal edge_attr must be 2D"
        assert graph[temporal_edge].edge_attr.shape[1] == config.TEMPORAL_EDGE_DIM, (
            f"Temporal edge_attr dim mismatch: got {graph[temporal_edge].edge_attr.shape[1]}, "
            f"expected {config.TEMPORAL_EDGE_DIM}"
        )

    for nt in ["file", "registry", "process", "network"]:
        if nt in graph.node_types and hasattr(graph[nt], "x"):
            assert graph[nt].x.ndim == 2, f"{nt}.x must be 2D"


def validate_api_timestamps(graph) -> tuple[bool, bool, int]:
    """
    Returns:
        has_timestamps, good_length, used_fallback
    """
    if "api" not in graph.node_types:
        return False, False, 0

    if not hasattr(graph["api"], "timestamps"):
        return False, False, 0

    ts = graph["api"].timestamps
    if ts is None:
        return False, False, 0

    good_length = hasattr(graph["api"], "x") and len(ts) == graph["api"].x.shape[0]

    used_fallback = 0
    if hasattr(graph["api"], "used_timestamp_fallback"):
        try:
            used_fallback = int(graph["api"].used_timestamp_fallback.item())
        except Exception:
            used_fallback = 0

    return True, good_length, used_fallback


def main() -> None:
    print("=" * 80)
    print("DATASET SMOKE TEST")
    print("=" * 80)
    print(f"ACTIVE_GRAPH_SOURCE: {config.ACTIVE_GRAPH_SOURCE}")
    print(f"ACTIVE_GRAPH_DIR   : {config.ACTIVE_GRAPH_DIR}")
    print(f"METADATA_PATH      : {config.METADATA_PATH}")
    print(f"NORMALIZE_NODE_FEATURE_DIMS: {config.NORMALIZE_NODE_FEATURE_DIMS}")
    print(f"FAIL_ON_SCHEMA_MISMATCH   : {config.FAIL_ON_SCHEMA_MISMATCH}")
    print()

    if not Path(config.ACTIVE_GRAPH_DIR).exists():
        raise FileNotFoundError(f"ACTIVE_GRAPH_DIR not found: {config.ACTIVE_GRAPH_DIR}")

    if not Path(config.METADATA_PATH).exists():
        raise FileNotFoundError(f"METADATA_PATH not found: {config.METADATA_PATH}")

    schema = ensure_active_schema()
    print("Schema loaded successfully.")
    print()

    # ------------------------------------------------------------
    # 1) Dataset construction
    # ------------------------------------------------------------
    datasets = {}
    for split in ["train", "val", "test"]:
        ds = RansomwareGraphDataset(
            data_dir="data",
            split=split,
            cache_dir=str(config.ACTIVE_GRAPH_DIR),
            schema=schema,
            validate_graphs=True,
        )
        datasets[split] = ds
        stats = ds.get_statistics()

        print(f"[{split.upper()}] Dataset loaded")
        print(f"  num_graphs       : {stats['num_graphs']}")
        print(f"  num_ransomware   : {stats['num_ransomware']}")
        print(f"  num_benign       : {stats['num_benign']}")
        print(f"  num_other_malware: {stats['num_other_malware']}")
        print(f"  graph_root       : {stats['graph_root']}")
        print()

    # ------------------------------------------------------------
    # 2) Single-item sampling + cleaning checks
    # ------------------------------------------------------------
    print("=" * 80)
    print("SINGLE GRAPH CHECKS")
    print("=" * 80)

    for split, ds in datasets.items():
        print(f"\n[{split.upper()}]")
        n_to_check = min(3, len(ds))

        for i in range(n_to_check):
            graph, label = ds[i]

            print(f"  sample {i}: label={label}")
            print(f"    node_types: {graph.node_types}")
            print(f"    edge_types: {graph.edge_types}")

            validate_single_graph_structure(graph)

            has_ts, good_ts_len, used_fb = validate_api_timestamps(graph)
            print(f"    api.timestamps present: {has_ts}")
            print(f"    api.timestamps good length: {good_ts_len}")
            print(f"    used_timestamp_fallback: {used_fb}")

            assert has_ts, "api.timestamps missing after dataset loading"
            assert good_ts_len, "api.timestamps length does not match api node count"

            for nt in ["file", "registry", "process", "network"]:
                if nt in graph.node_types and hasattr(graph[nt], "x"):
                    print(f"    {nt}.x shape: {tuple(graph[nt].x.shape)}")

            print(f"    api.x shape: {tuple(graph['api'].x.shape)}")
            print(f"    temporal edge count: {graph[('api', 'temporal', 'api')].edge_index.shape[1]}")

            print_suspicious_keys(graph, "    BEFORE CLEAN")
            cleaned = _clean_graph_for_batching(graph)
            print_suspicious_keys(cleaned, "    AFTER CLEAN")
            assert_clean_for_batching(cleaned)

    # ------------------------------------------------------------
    # 3) Timestamp fallback audit over a sample of each split
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TIMESTAMP FALLBACK CHECK")
    print("=" * 80)

    for split, ds in datasets.items():
        checked = min(200, len(ds))
        fallback_used = 0
        timestamps_missing_after = 0
        bad_len_after = 0

        for i in range(checked):
            graph, _ = ds[i]
            has_ts, good_ts_len, used_fb = validate_api_timestamps(graph)

            if not has_ts:
                timestamps_missing_after += 1
            if has_ts and not good_ts_len:
                bad_len_after += 1
            fallback_used += used_fb

        print(
            f"{split}: checked={checked} | "
            f"fallback_used={fallback_used} | "
            f"timestamps_missing_after={timestamps_missing_after} | "
            f"bad_len_after={bad_len_after}"
        )

        assert timestamps_missing_after == 0, (
            f"{split}: some graphs still missing api.timestamps after dataset loading"
        )
        assert bad_len_after == 0, (
            f"{split}: some graphs have bad api.timestamps length after dataset loading"
        )

    # ------------------------------------------------------------
    # 4) Manual mini-batch collation stress test
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MANUAL MINI-BATCH CLEANING CHECK")
    print("=" * 80)

    train_ds = datasets["train"]
    n_manual = min(8, len(train_ds))
    manual_graphs = []
    manual_labels = []
    for i in range(n_manual):
        g, y = train_ds[i]
        g = _clean_graph_for_batching(g)
        assert_clean_for_batching(g)
        manual_graphs.append(g)
        manual_labels.append(y)

    from torch_geometric.data import Batch
    manual_batch = Batch.from_data_list(manual_graphs)
    print(f"Manual batch built successfully with {len(manual_graphs)} graphs")
    print(f"Manual batch api.x shape: {tuple(manual_batch['api'].x.shape)}")
    print(f"Manual labels: {manual_labels}")

    # ------------------------------------------------------------
    # 5) DataLoader / batching smoke test on multiple batches
    # ------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BATCHING CHECKS")
    print("=" * 80)

    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir="data",
        cache_dir=str(config.ACTIVE_GRAPH_DIR),
        batch_size=4,
        num_workers=0,
        shuffle_train=False,
        schema=schema,
    )

    loaders = [("train", train_loader), ("val", val_loader), ("test", test_loader)]

    for name, loader in loaders:
        print(f"\n[{name.upper()} BATCHES]")
        max_batches = 5 if name == "train" else 3

        for batch_idx, (batch, labels) in enumerate(loader):
            if batch_idx >= max_batches:
                break

            print(f"  batch {batch_idx}")
            print(f"    batch type     : {type(batch)}")
            print(f"    num graphs     : {batch.num_graphs}")
            print(f"    labels shape   : {tuple(labels.shape)}")
            print(f"    labels         : {labels.tolist()}")

            assert isinstance(labels, torch.Tensor), "labels must be a tensor"
            assert labels.ndim == 1, "labels must be 1D"
            assert labels.shape[0] == batch.num_graphs, "labels size must match batch.num_graphs"

            assert "api" in batch.node_types, "Batched graph missing 'api' node type"
            assert hasattr(batch["api"], "x"), "Batched api.x missing"
            assert batch["api"].x.ndim == 2, "Batched api.x must be 2D"
            assert batch["api"].x.shape[1] == config.API_NODE_DIM, (
                f"Batched api.x dim mismatch: got {batch['api'].x.shape[1]}, "
                f"expected {config.API_NODE_DIM}"
            )

            temporal_edge = ("api", "temporal", "api")
            assert temporal_edge in batch.edge_types, "Batched graph missing temporal edge type"
            if hasattr(batch[temporal_edge], "edge_attr"):
                assert batch[temporal_edge].edge_attr.ndim == 2, "Batched temporal edge_attr must be 2D"
                assert batch[temporal_edge].edge_attr.shape[1] == config.TEMPORAL_EDGE_DIM, (
                    f"Batched temporal edge_attr dim mismatch: got {batch[temporal_edge].edge_attr.shape[1]}, "
                    f"expected {config.TEMPORAL_EDGE_DIM}"
                )

            print(f"    api.x shape    : {tuple(batch['api'].x.shape)}")
            print(f"    temporal edges : {batch[temporal_edge].edge_index.shape[1]}")
            if hasattr(batch[temporal_edge], "edge_attr"):
                print(f"    temporal attr  : {tuple(batch[temporal_edge].edge_attr.shape)}")

            for nt in batch.node_types:
                if hasattr(batch[nt], "x"):
                    print(f"    {nt}.x shape    : {tuple(batch[nt].x.shape)}")

    print("\n✅ DATASET SMOKE TEST PASSED")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n❌ DATASET SMOKE TEST FAILED")
        print(f"Error: {e}")
        traceback.print_exc()
        raise