"""
utils/schema_manager.py

Schema lifecycle manager for the active graph source.

Purpose:
- Ensure the active schema exists
- Detect when the schema is stale relative to the active graph root
- Regenerate schema automatically by running SchemaInspector if needed
- Load and return the active schema for training/inference

Typical use:
    from utils.schema_manager import ensure_active_schema

    schema = ensure_active_schema()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import config
from utils.schema_inspector import SchemaInspector
from utils.schema_utils import load_schema, summarize_schema


def _latest_mtime_under(root: Path, pattern: str = "*.pkl") -> Optional[float]:
    """
    Return latest modification time under root matching pattern.
    Returns None if no matching files are found.
    """
    if not root.exists():
        return None

    latest = None
    for path in root.rglob(pattern):
        try:
            mtime = path.stat().st_mtime
            if latest is None or mtime > latest:
                latest = mtime
        except FileNotFoundError:
            continue

    return latest


def schema_exists(schema_path: Optional[Path] = None) -> bool:
    """Check whether active schema JSON exists."""
    path = Path(schema_path) if schema_path else config.ACTIVE_SCHEMA_PATH
    return path.exists()


def is_schema_stale(
    graph_root: Optional[Path] = None,
    schema_path: Optional[Path] = None,
) -> bool:
    """
    Determine whether schema is stale relative to graph pickle files.

    Policy:
    - If schema does not exist -> stale
    - If no graph files exist -> not stale (caller can handle separately)
    - If any graph file is newer than schema file -> stale
    """
    graph_root = Path(graph_root) if graph_root else config.ACTIVE_GRAPH_DIR
    schema_path = Path(schema_path) if schema_path else config.ACTIVE_SCHEMA_PATH

    if not schema_path.exists():
        return True

    schema_mtime = schema_path.stat().st_mtime
    latest_graph_mtime = _latest_mtime_under(graph_root, "*.pkl")

    if latest_graph_mtime is None:
        return False

    return latest_graph_mtime > schema_mtime


def regenerate_active_schema(
    graph_root: Optional[Path] = None,
    schema_path: Optional[Path] = None,
    schema_summary_path: Optional[Path] = None,
    source_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Regenerate schema for the active graph source using SchemaInspector.
    """
    graph_root = Path(graph_root) if graph_root else config.ACTIVE_GRAPH_DIR
    schema_path = Path(schema_path) if schema_path else config.ACTIVE_SCHEMA_PATH
    schema_summary_path = (
        Path(schema_summary_path)
        if schema_summary_path
        else config.ACTIVE_SCHEMA_SUMMARY_PATH
    )
    source_name = source_name or config.ACTIVE_GRAPH_SOURCE

    print(f"\n[SchemaManager] Regenerating schema for source: {source_name}")
    print(f"[SchemaManager] Graph root: {graph_root}")
    print(f"[SchemaManager] Schema JSON: {schema_path}")

    inspector = SchemaInspector(
        graph_root=graph_root,
        output_json=schema_path,
        output_summary=schema_summary_path,
        source_name=source_name,
    )
    return inspector.inspect()


def ensure_active_schema(
    auto_refresh: Optional[bool] = None,
    graph_root: Optional[Path] = None,
    schema_path: Optional[Path] = None,
    schema_summary_path: Optional[Path] = None,
    source_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ensure the active schema exists and is current.

    Behavior:
    - If schema missing -> regenerate
    - If schema stale and AUTO_REFRESH_SCHEMA=True -> regenerate
    - If schema stale and AUTO_REFRESH_SCHEMA=False -> raise
    - Else -> load existing schema

    Returns:
        Loaded schema dict
    """
    auto_refresh = config.AUTO_REFRESH_SCHEMA if auto_refresh is None else auto_refresh
    graph_root = Path(graph_root) if graph_root else config.ACTIVE_GRAPH_DIR
    schema_path = Path(schema_path) if schema_path else config.ACTIVE_SCHEMA_PATH
    schema_summary_path = (
        Path(schema_summary_path)
        if schema_summary_path
        else config.ACTIVE_SCHEMA_SUMMARY_PATH
    )
    source_name = source_name or config.ACTIVE_GRAPH_SOURCE

    if not graph_root.exists():
        raise FileNotFoundError(
            f"Active graph root does not exist: {graph_root}\n"
            f"Check config.ACTIVE_GRAPH_SOURCE / config.GRAPH_SOURCES."
        )

    needs_refresh = (not schema_exists(schema_path)) or is_schema_stale(graph_root, schema_path)

    if needs_refresh:
        if auto_refresh:
            regenerate_active_schema(
                graph_root=graph_root,
                schema_path=schema_path,
                schema_summary_path=schema_summary_path,
                source_name=source_name,
            )
        else:
            raise RuntimeError(
                f"Schema is missing or stale for source '{source_name}'.\n"
                f"Schema path: {schema_path}\n"
                f"Set AUTO_REFRESH_SCHEMA=True or rerun schema_inspector manually."
            )

    schema = load_schema(schema_path)

    print("\n[SchemaManager] Active schema ready.")
    summary = summarize_schema(schema)
    print(f"[SchemaManager] Source       : {summary['source_name']}")
    print(f"[SchemaManager] Graph root   : {summary['graph_root']}")
    print(f"[SchemaManager] Node types   : {summary['node_types']}")
    print(f"[SchemaManager] Edge types   : {summary['edge_types']}")

    return schema


def print_active_schema_status(
    graph_root: Optional[Path] = None,
    schema_path: Optional[Path] = None,
    source_name: Optional[str] = None,
) -> None:
    """
    Lightweight debug/status printer.
    """
    graph_root = Path(graph_root) if graph_root else config.ACTIVE_GRAPH_DIR
    schema_path = Path(schema_path) if schema_path else config.ACTIVE_SCHEMA_PATH
    source_name = source_name or config.ACTIVE_GRAPH_SOURCE

    exists = schema_exists(schema_path)
    stale = is_schema_stale(graph_root, schema_path) if exists else True

    print("\n[SchemaManager] Status")
    print(f"  Source      : {source_name}")
    print(f"  Graph root  : {graph_root}")
    print(f"  Schema path : {schema_path}")
    print(f"  Exists      : {exists}")
    print(f"  Stale       : {stale}")