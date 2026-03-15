"""
utils/schema_utils.py

Helpers for loading, interpreting, and applying the active graph schema.

Purpose:
- Load schema.json for the currently active graph source
- Expose canonical node/edge type ordering
- Expose canonical per-node-type feature dimensions
- Normalize graph node features to schema-defined dimensions
- Provide lightweight schema query helpers for dataset/model/validator code
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

import config


EdgeType = Tuple[str, str, str]


def edge_type_to_str(edge_type: EdgeType) -> str:
    """Convert edge type tuple to canonical string form."""
    src, rel, dst = edge_type
    return f"{src}::{rel}::{dst}"


def str_to_edge_type(edge_type_str: str) -> EdgeType:
    """Convert canonical edge type string back to tuple form."""
    parts = edge_type_str.split("::")
    if len(parts) != 3:
        raise ValueError(f"Invalid edge type string: {edge_type_str}")
    return parts[0], parts[1], parts[2]


def load_schema(schema_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load schema.json from the active schema path unless explicitly provided.
    """
    path = Path(schema_path) if schema_path else config.ACTIVE_SCHEMA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Schema file not found: {path}\n"
            f"Run utils/schema_inspector.py first, or enable schema auto-refresh."
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_active_schema() -> Dict[str, Any]:
    """Convenience helper for current active schema."""
    return load_schema(config.ACTIVE_SCHEMA_PATH)


def get_active_source_name(schema: Optional[Dict[str, Any]] = None) -> str:
    """Return active schema source name."""
    schema = schema or get_active_schema()
    return schema["source"]["name"]


def get_schema_graph_root(schema: Optional[Dict[str, Any]] = None) -> Path:
    """Return graph root path recorded in schema."""
    schema = schema or get_active_schema()
    return Path(schema["source"]["graph_root"])


def get_node_types(schema: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Return stable node type order for current schema.

    Prefers schema-found order aligned to config fallback order, then appends
    any unexpected node types in sorted order.
    """
    schema = schema or get_active_schema()

    found = schema.get("schema_summary", {}).get("node_types_found", [])
    fallback = list(getattr(config, "DEFAULT_NODE_TYPE_ORDER", []))

    ordered = [nt for nt in fallback if nt in found]
    extras = sorted([nt for nt in found if nt not in ordered])

    return ordered + extras


def get_edge_types(schema: Optional[Dict[str, Any]] = None) -> List[EdgeType]:
    """
    Return stable edge type order for current schema.

    Prefers config fallback order, then appends additional discovered edge types.
    """
    schema = schema or get_active_schema()

    found_strs = schema.get("schema_summary", {}).get("edge_types_found", [])
    fallback_tuples = list(getattr(config, "DEFAULT_EDGE_TYPE_ORDER", []))
    fallback_strs = [edge_type_to_str(et) for et in fallback_tuples]

    ordered_strs = [et for et in fallback_strs if et in found_strs]
    extras = sorted([et for et in found_strs if et not in ordered_strs])

    return [str_to_edge_type(et) for et in (ordered_strs + extras)]


def get_node_feature_dims(schema: Optional[Dict[str, Any]] = None) -> Dict[str, List[int]]:
    """
    Return all observed raw feature dims per node type from schema.
    Example:
        {"api": [775], "file": [6, 7], ...}
    """
    schema = schema or get_active_schema()

    out: Dict[str, List[int]] = {}
    for node_type, stats in schema.get("node_types", {}).items():
        out[node_type] = list(stats.get("feature_dims") or [])
    return out


def get_edge_feature_dims(schema: Optional[Dict[str, Any]] = None) -> Dict[EdgeType, List[int]]:
    """
    Return all observed edge_attr dims per edge type from schema.
    """
    schema = schema or get_active_schema()

    out: Dict[EdgeType, List[int]] = {}
    for edge_type_str, stats in schema.get("edge_types", {}).items():
        out[str_to_edge_type(edge_type_str)] = list(stats.get("edge_attr_dims") or [])
    return out


def get_canonical_node_dims(schema: Optional[Dict[str, Any]] = None) -> Dict[str, int]:
    """
    Choose canonical raw dim per node type for the current schema.

    Policy:
    - If exactly one dim is observed, use it
    - If multiple dims are observed, use the maximum observed dim
    - This preserves all observed dimensions by allowing right-padding in loader
    """
    schema = schema or get_active_schema()

    dims_by_type = get_node_feature_dims(schema)
    canonical: Dict[str, int] = {}

    for node_type, dims in dims_by_type.items():
        if not dims:
            # Fallbacks for cases with no explicit x dim observed
            if node_type == "api":
                canonical[node_type] = getattr(config, "API_NODE_DIM", 775)
            else:
                canonical[node_type] = getattr(config, "RESOURCE_NODE_DIM", 6)
        elif len(dims) == 1:
            canonical[node_type] = dims[0]
        else:
            canonical[node_type] = max(dims)

    return canonical


def get_canonical_edge_dims(schema: Optional[Dict[str, Any]] = None) -> Dict[EdgeType, Optional[int]]:
    """
    Choose canonical edge_attr dim per edge type.

    Policy:
    - If no edge_attr dims observed, return None
    - If one or more observed, use max observed
    """
    schema = schema or get_active_schema()

    dims_by_type = get_edge_feature_dims(schema)
    canonical: Dict[EdgeType, Optional[int]] = {}

    for edge_type, dims in dims_by_type.items():
        canonical[edge_type] = max(dims) if dims else None

    return canonical


def get_nonempty_edge_types(schema: Optional[Dict[str, Any]] = None) -> List[EdgeType]:
    """
    Return only relations that have at least one edge in at least one graph.
    """
    schema = schema or get_active_schema()

    nonempty: List[EdgeType] = []
    for edge_type_str, stats in schema.get("edge_types", {}).items():
        total_edges = int(stats.get("total_edges", 0))
        if total_edges > 0:
            nonempty.append(str_to_edge_type(edge_type_str))

    # Preserve stable order
    ordered = get_edge_types(schema)
    nonempty_set = set(nonempty)
    return [et for et in ordered if et in nonempty_set]


def get_registered_but_empty_edge_types(schema: Optional[Dict[str, Any]] = None) -> List[EdgeType]:
    """
    Return relations present in schema but with zero total edges across dataset.
    """
    schema = schema or get_active_schema()

    empty: List[EdgeType] = []
    for edge_type_str, stats in schema.get("edge_types", {}).items():
        total_edges = int(stats.get("total_edges", 0))
        if total_edges == 0:
            empty.append(str_to_edge_type(edge_type_str))

    ordered = get_edge_types(schema)
    empty_set = set(empty)
    return [et for et in ordered if et in empty_set]


def get_missing_expected_node_types(schema: Optional[Dict[str, Any]] = None) -> List[str]:
    schema = schema or get_active_schema()
    return list(schema.get("compatibility_checks", {}).get("missing_expected_node_types", []))


def get_missing_expected_edge_types(schema: Optional[Dict[str, Any]] = None) -> List[EdgeType]:
    schema = schema or get_active_schema()
    missing = schema.get("compatibility_checks", {}).get("missing_expected_edge_types", [])
    return [str_to_edge_type(et) for et in missing]


def pad_or_validate_features(
    x: torch.Tensor,
    target_dim: int,
    node_type: str,
    allow_truncation: Optional[bool] = None,
) -> torch.Tensor:
    """
    Normalize a node feature matrix to the target dim.

    Policy:
    - if current dim == target_dim: unchanged
    - if current dim < target_dim: right-pad zeros
    - if current dim > target_dim:
        - truncate only if allow_truncation=True
        - else raise ValueError
    """
    if x.dim() != 2:
        raise ValueError(
            f"Expected 2D feature tensor for node type '{node_type}', got shape={tuple(x.shape)}"
        )

    allow_truncation = (
        config.ALLOW_FEATURE_TRUNCATION if allow_truncation is None else allow_truncation
    )

    current_dim = int(x.size(1))

    if current_dim == target_dim:
        return x

    if current_dim < target_dim:
        pad_cols = target_dim - current_dim
        pad = torch.zeros(
            (x.size(0), pad_cols),
            dtype=x.dtype,
            device=x.device,
        )
        return torch.cat([x, pad], dim=1)

    # current_dim > target_dim
    if allow_truncation:
        return x[:, :target_dim]

    raise ValueError(
        f"Feature dim for node type '{node_type}' exceeds canonical dim: "
        f"current_dim={current_dim}, target_dim={target_dim}. "
        f"Set ALLOW_FEATURE_TRUNCATION=True only if this is intentional."
    )


def normalize_graph_node_features(
    graph: Any,
    schema: Optional[Dict[str, Any]] = None,
    inplace: bool = True,
) -> Any:
    """
    Normalize node feature dimensions in a HeteroData graph to canonical schema dims.

    This does NOT remove node types or relations.
    It only makes node feature matrices shape-consistent.
    """
    schema = schema or get_active_schema()
    canonical_dims = get_canonical_node_dims(schema)

    data = graph if inplace else graph.clone()

    for node_type in getattr(data, "node_types", []):
        if node_type not in canonical_dims:
            # Unknown type in graph that is not in schema; leave untouched.
            continue

        store = data[node_type]
        x = getattr(store, "x", None)
        if x is None:
            continue

        target_dim = canonical_dims[node_type]
        store.x = pad_or_validate_features(x, target_dim, node_type)

    return data


def summarize_schema(schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Lightweight summary used by train.py / logging.
    """
    schema = schema or get_active_schema()

    return {
        "source_name": get_active_source_name(schema),
        "graph_root": str(get_schema_graph_root(schema)),
        "node_types": get_node_types(schema),
        "edge_types": [edge_type_to_str(et) for et in get_edge_types(schema)],
        "canonical_node_dims": get_canonical_node_dims(schema),
        "canonical_edge_dims": {
            edge_type_to_str(k): v for k, v in get_canonical_edge_dims(schema).items()
        },
        "nonempty_edge_types": [edge_type_to_str(et) for et in get_nonempty_edge_types(schema)],
        "empty_edge_types": [
            edge_type_to_str(et) for et in get_registered_but_empty_edge_types(schema)
        ],
    }