# scripts/check_dataset.py
import pickle
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

enriched_root = PROJECT_ROOT / "cache" / "enriched"

# ── grab one PKL from any split ───────────────────────────────────
for split in ["train", "val", "test"]:
    pkls = list((enriched_root / split).glob("*.pkl"))
    if pkls:
        print(f"Found {len(pkls)} files in '{split}' — loading first one...")
        graph = pickle.load(open(pkls[0], "rb"))
        break

# ── inspect structure ─────────────────────────────────────────────
print("\nType of graph object :", type(graph))

if hasattr(graph, 'node_types'):
    print("Node types           :", graph.node_types)
    print("Edge types           :", graph.edge_types)
    for nt in graph.node_types:
        node = graph[nt]
        print(f"\n  [{nt}]")
        print(f"    keys/attrs : {list(node.__dict__.keys())}")
        if hasattr(node, 'x'):
            print(f"    x.shape    : {node.x.shape}")
        if hasattr(node, 'api_names'):
            print(f"    api_names  : {node.api_names[:5]}")
        if hasattr(node, 'node_names'):
            print(f"    node_names : {node.node_names[:5]}")

elif isinstance(graph, tuple):
    print("Tuple length         :", len(graph))
    for i, item in enumerate(graph):
        print(f"\n  [{i}] type: {type(item)}")
        if hasattr(item, 'node_types'):
            print(f"    node_types : {item.node_types}")
            for nt in item.node_types:
                node = item[nt]
                print(f"    [{nt}] attrs: {list(node.__dict__.keys())}")

elif isinstance(graph, dict):
    print("Top-level keys       :", list(graph.keys()))
