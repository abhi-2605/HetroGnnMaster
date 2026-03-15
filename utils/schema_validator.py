"""
utils/schema_validator.py

Validation helpers for schema-aware graph loading and training.

Purpose:
- Validate a single HeteroData graph against the active schema
- Validate basic batch-level assumptions before model forward
- Fail early with clear errors instead of silent shape mismatches

Usage:
    from utils.schema_validator import validate_graph, validate_graph_or_raise
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

import config
from utils.schema_utils import (
    get_active_schema,
    get_node_types,
    get_edge_types,
    get_canonical_node_dims,
    get_canonical_edge_dims,
    get_nonempty_edge_types,
    edge_type_to_str,
)


EdgeType = Tuple[str, str, str]


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)


def _has_attr(obj: Any, attr: str) -> bool:
    return hasattr(obj, attr) and getattr(obj, attr) is not None


def _get_num_nodes(store: Any) -> int:
    if _has_attr(store, "x"):
        x = getattr(store, "x")
        if isinstance(x, torch.Tensor) and x.dim() >= 1:
            return int(x.size(0))
    return int(getattr(store, "num_nodes", 0) or 0)


def _get_feat_dim(store: Any) -> Optional[int]:
    if _has_attr(store, "x"):
        x = getattr(store, "x")
        if isinstance(x, torch.Tensor) and x.dim() == 2:
            return int(x.size(1))
    return None


def _get_num_edges(store: Any) -> int:
    if _has_attr(store, "edge_index"):
        edge_index = getattr(store, "edge_index")
        if isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2:
            return int(edge_index.size(1))
    return 0


def _get_edge_attr_dim(store: Any) -> Optional[int]:
    if _has_attr(store, "edge_attr"):
        edge_attr = getattr(store, "edge_attr")
        if isinstance(edge_attr, torch.Tensor):
            if edge_attr.dim() == 2:
                return int(edge_attr.size(1))
            if edge_attr.dim() == 1:
                return 1
    return None


def validate_graph(
    graph: Any,
    schema: Optional[Dict[str, Any]] = None,
    require_api_nodes: bool = True,
    strict: Optional[bool] = None,
) -> ValidationResult:
    """
    Validate a single HeteroData graph against the active schema.

    Checks:
    - node types against schema
    - canonical node feature dims
    - edge types against schema
    - canonical edge_attr dims where applicable
    - temporal edge_attr presence/dim when relation exists and is non-empty
    - API node existence when required
    """
    schema = schema or get_active_schema()
    strict = config.SCHEMA_STRICT_VALIDATION if strict is None else strict

    result = ValidationResult(is_valid=True)

    expected_node_types = set(get_node_types(schema))
    expected_edge_types = set(get_edge_types(schema))
    canonical_node_dims = get_canonical_node_dims(schema)
    canonical_edge_dims = get_canonical_edge_dims(schema)
    nonempty_edge_types = set(get_nonempty_edge_types(schema))

    graph_node_types = set(getattr(graph, "node_types", []))
    graph_edge_types = set(getattr(graph, "edge_types", []))

    result.info["graph_node_types"] = sorted(list(graph_node_types))
    result.info["graph_edge_types"] = [edge_type_to_str(et) for et in sorted(graph_edge_types)]

    missing_node_types = expected_node_types - graph_node_types
    unexpected_node_types = graph_node_types - expected_node_types

    if missing_node_types:
        result.warnings.append(
            f"Missing node types in graph: {sorted(list(missing_node_types))}"
        )

    if unexpected_node_types:
        msg = f"Unexpected node types in graph: {sorted(list(unexpected_node_types))}"
        if strict:
            result.errors.append(msg)
        else:
            result.warnings.append(msg)

    # Validate node stores
    for node_type in graph_node_types:
        store = graph[node_type]

        num_nodes = _get_num_nodes(store)
        feat_dim = _get_feat_dim(store)

        if require_api_nodes and node_type == "api" and num_nodes == 0:
            result.errors.append("Graph has zero API nodes; API branch cannot operate meaningfully.")

        if feat_dim is None and num_nodes > 0:
            result.errors.append(
                f"Node type '{node_type}' has nodes but no valid 2D feature matrix x."
            )
            continue

        if feat_dim is not None:
            expected_dim = canonical_node_dims.get(node_type)
            if expected_dim is not None and feat_dim != expected_dim:
                result.errors.append(
                    f"Node type '{node_type}' feature dim mismatch: "
                    f"got {feat_dim}, expected {expected_dim}."
                )

    # Validate edge stores
    missing_edge_types = expected_edge_types - graph_edge_types
    unexpected_edge_types = graph_edge_types - expected_edge_types

    if missing_edge_types:
        result.warnings.append(
            "Missing edge types in graph: "
            f"{[edge_type_to_str(et) for et in sorted(missing_edge_types)]}"
        )

    if unexpected_edge_types:
        msg = (
            "Unexpected edge types in graph: "
            f"{[edge_type_to_str(et) for et in sorted(unexpected_edge_types)]}"
        )
        if strict:
            result.errors.append(msg)
        else:
            result.warnings.append(msg)

    for edge_type in graph_edge_types:
        store = graph[edge_type]
        num_edges = _get_num_edges(store)
        edge_attr_dim = _get_edge_attr_dim(store)
        expected_edge_dim = canonical_edge_dims.get(edge_type)

        # If relation is known to carry edge attrs in schema and is non-empty,
        # validate edge_attr dimensionality
        if edge_type in nonempty_edge_types and expected_edge_dim is not None:
            if num_edges > 0 and edge_attr_dim is None:
                result.errors.append(
                    f"Edge type '{edge_type_to_str(edge_type)}' has edges but no edge_attr."
                )
            elif num_edges > 0 and edge_attr_dim != expected_edge_dim:
                result.errors.append(
                    f"Edge type '{edge_type_to_str(edge_type)}' edge_attr dim mismatch: "
                    f"got {edge_attr_dim}, expected {expected_edge_dim}."
                )

        # Special check for temporal edge consistency
        if edge_type == ("api", "temporal", "api") and num_edges > 0:
            expected_temporal_dim = getattr(config, "TEMPORAL_EDGE_DIM", None)
            if expected_temporal_dim is not None:
                if edge_attr_dim is None:
                    result.errors.append(
                        "Temporal API edges exist but edge_attr is missing."
                    )
                elif edge_attr_dim != expected_temporal_dim:
                    result.errors.append(
                        f"Temporal API edge_attr dim mismatch: got {edge_attr_dim}, "
                        f"expected {expected_temporal_dim}."
                    )

    result.is_valid = len(result.errors) == 0
    return result


def validate_graph_or_raise(
    graph: Any,
    schema: Optional[Dict[str, Any]] = None,
    require_api_nodes: bool = True,
    strict: Optional[bool] = None,
) -> None:
    """
    Validate graph and raise ValueError if invalid.
    """
    result = validate_graph(
        graph=graph,
        schema=schema,
        require_api_nodes=require_api_nodes,
        strict=strict,
    )

    if not result.is_valid:
        message = ["Graph validation failed:"]
        message.extend([f"  - {err}" for err in result.errors])
        if result.warnings:
            message.append("Warnings:")
            message.extend([f"  - {warn}" for warn in result.warnings])
        raise ValueError("\n".join(message))


def validate_batch_dicts(
    x_dict: Dict[str, torch.Tensor],
    edge_index_dict: Dict[EdgeType, torch.Tensor],
    edge_attr_dict: Optional[Dict[EdgeType, torch.Tensor]] = None,
    batch_dict: Optional[Dict[str, torch.Tensor]] = None,
    schema: Optional[Dict[str, Any]] = None,
    require_api: bool = True,
    require_api_indices_if_memory: bool = False,
    api_indices: Optional[torch.Tensor] = None,
) -> ValidationResult:
    """
    Validate model-ready batch dictionaries before forward pass.

    Useful in train.py just before model(...) to catch shape issues early.
    """
    schema = schema or get_active_schema()
    result = ValidationResult(is_valid=True)

    canonical_node_dims = get_canonical_node_dims(schema)
    canonical_edge_dims = get_canonical_edge_dims(schema)
    nonempty_edge_types = set(get_nonempty_edge_types(schema))

    # Validate x_dict
    for node_type, x in x_dict.items():
        if not isinstance(x, torch.Tensor):
            result.errors.append(f"x_dict['{node_type}'] is not a tensor.")
            continue

        if x.dim() != 2:
            result.errors.append(
                f"x_dict['{node_type}'] must be 2D, got shape={tuple(x.shape)}."
            )
            continue

        expected_dim = canonical_node_dims.get(node_type)
        if expected_dim is not None and int(x.size(1)) != expected_dim:
            result.errors.append(
                f"x_dict['{node_type}'] feature dim mismatch: "
                f"got {int(x.size(1))}, expected {expected_dim}."
            )

        if require_api and node_type == "api" and int(x.size(0)) == 0:
            result.errors.append("Batch has zero API nodes.")

    # Validate edge_index_dict
    for edge_type, edge_index in edge_index_dict.items():
        if not isinstance(edge_index, torch.Tensor):
            result.errors.append(
                f"edge_index_dict['{edge_type_to_str(edge_type)}'] is not a tensor."
            )
            continue

        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            result.errors.append(
                f"edge_index_dict['{edge_type_to_str(edge_type)}'] must have shape [2, E], "
                f"got {tuple(edge_index.shape)}."
            )

        if edge_type in nonempty_edge_types:
            expected_edge_dim = canonical_edge_dims.get(edge_type)
            if expected_edge_dim is not None:
                if edge_attr_dict is None or edge_type not in edge_attr_dict:
                    if edge_index.size(1) > 0:
                        result.errors.append(
                            f"Missing edge_attr for non-empty relation "
                            f"'{edge_type_to_str(edge_type)}'."
                        )
                else:
                    edge_attr = edge_attr_dict[edge_type]
                    if edge_attr.dim() not in (1, 2):
                        result.errors.append(
                            f"edge_attr_dict['{edge_type_to_str(edge_type)}'] has invalid shape "
                            f"{tuple(edge_attr.shape)}."
                        )
                    else:
                        got_dim = int(edge_attr.size(1)) if edge_attr.dim() == 2 else 1
                        if got_dim != expected_edge_dim:
                            result.errors.append(
                                f"edge_attr dim mismatch for relation "
                                f"'{edge_type_to_str(edge_type)}': got {got_dim}, "
                                f"expected {expected_edge_dim}."
                            )

    # Validate batch_dict
    if batch_dict is not None:
        for node_type, x in x_dict.items():
            if node_type not in batch_dict:
                result.warnings.append(f"Missing batch vector for node type '{node_type}'.")
                continue

            batch_vec = batch_dict[node_type]
            if not isinstance(batch_vec, torch.Tensor):
                result.errors.append(f"batch_dict['{node_type}'] is not a tensor.")
                continue

            if batch_vec.dim() != 1:
                result.errors.append(
                    f"batch_dict['{node_type}'] must be 1D, got shape={tuple(batch_vec.shape)}."
                )
                continue

            if int(batch_vec.size(0)) != int(x.size(0)):
                result.errors.append(
                    f"batch_dict['{node_type}'] length mismatch: "
                    f"got {int(batch_vec.size(0))}, expected {int(x.size(0))}."
                )

    # Validate memory requirements
    if require_api_indices_if_memory and getattr(config, "USE_MEMORY", False):
        if api_indices is None:
            result.errors.append(
                "Memory is enabled but api_indices is missing from batch/model input."
            )
        elif not isinstance(api_indices, torch.Tensor):
            result.errors.append("api_indices is not a tensor.")
        elif api_indices.dim() != 1:
            result.errors.append(
                f"api_indices must be 1D, got shape={tuple(api_indices.shape)}."
            )
        elif "api" in x_dict and int(api_indices.size(0)) != int(x_dict["api"].size(0)):
            result.errors.append(
                "api_indices length mismatch with number of API nodes: "
                f"got {int(api_indices.size(0))}, expected {int(x_dict['api'].size(0))}."
            )

    result.is_valid = len(result.errors) == 0
    return result


def validate_batch_or_raise(
    x_dict: Dict[str, torch.Tensor],
    edge_index_dict: Dict[EdgeType, torch.Tensor],
    edge_attr_dict: Optional[Dict[EdgeType, torch.Tensor]] = None,
    batch_dict: Optional[Dict[str, torch.Tensor]] = None,
    schema: Optional[Dict[str, Any]] = None,
    require_api: bool = True,
    require_api_indices_if_memory: bool = False,
    api_indices: Optional[torch.Tensor] = None,
) -> None:
    """
    Validate model-ready batch dicts and raise ValueError if invalid.
    """
    result = validate_batch_dicts(
        x_dict=x_dict,
        edge_index_dict=edge_index_dict,
        edge_attr_dict=edge_attr_dict,
        batch_dict=batch_dict,
        schema=schema,
        require_api=require_api,
        require_api_indices_if_memory=require_api_indices_if_memory,
        api_indices=api_indices,
    )

    if not result.is_valid:
        message = ["Batch validation failed:"]
        message.extend([f"  - {err}" for err in result.errors])
        if result.warnings:
            message.append("Warnings:")
            message.extend([f"  - {warn}" for warn in result.warnings])
        raise ValueError("\n".join(message))