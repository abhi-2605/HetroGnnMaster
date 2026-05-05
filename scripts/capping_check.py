from pathlib import Path
import pickle
import torch
import config

split = "test"
limit = 5

graph_dir = Path(config.ACTIVE_GRAPH_DIR) / split
cap = getattr(config, "BEHAVIOR_AWARE_API_CAP_MAX_NODES", 30000)
threshold = getattr(config, "BEHAVIOR_AWARE_API_CAP_THRESHOLD", 100000)

large_graphs = []

for p in graph_dir.rglob("*.pkl"):
    with open(p, "rb") as f:
        g = pickle.load(f)

    if hasattr(g, "x_dict") and "api" in g.x_dict:
        n_api = g.x_dict["api"].size(0)
    elif hasattr(g, "node_types") and "api" in g.node_types:
        n_api = g["api"].x.size(0)
    else:
        continue

    if n_api > threshold:
        large_graphs.append((p, n_api))

large_graphs.sort(key=lambda x: x[1], reverse=True)

print(f"Checking split: {split}")
print(f"Threshold     : {threshold}")
print(f"Cap           : {cap}")
print(f"Large graphs  : {len(large_graphs)}")

for p, n_api in large_graphs[:limit]:
    print("\n" + "=" * 80)
    print(f"Graph: {p.name}")
    print(f"Before API nodes: {n_api}")

    with open(p, "rb") as f:
        g = pickle.load(f)

    # simple small-scale cap simulation: keep first cap API nodes
    keep = torch.arange(cap)

    if hasattr(g, "x_dict") and "api" in g.x_dict:
        api_x = g.x_dict["api"]
        capped_api_x = api_x[keep]
    else:
        api_x = g["api"].x
        capped_api_x = api_x[keep]

    print(f"After API nodes : {capped_api_x.size(0)}")
    print("Dry run only    : original graph NOT modified")