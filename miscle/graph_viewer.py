# """Quick graph inspector"""
# import pickle
# from pathlib import Path
# import sys

# cache_dir = Path("../cache/train")
# graph_files = sorted(cache_dir.glob("*.pkl"))

# if not graph_files:
#     print("❌ No graphs found in cache/train/")
#     sys.exit(1)

# # Load first graph or specified index
# idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
# graph_file = graph_files[idx]

# print(f"📄 Loading: {graph_file.name}\n")

# with open(graph_file, 'rb') as f:
#     graph = pickle.load(f)

# print("="*60)
# print("GRAPH STRUCTURE")
# print("="*60)

# print(f"\n✓ Node Types ({len(graph.node_types)}):")
# for nt in graph.node_types:
#     print(f"  {nt:12} : {graph[nt].num_nodes:5} nodes")

# print(f"\n✓ Edge Types ({len(graph.edge_types)}):")
# for et in graph.edge_types:
#     print(f"  {str(et):50} : {graph[et].edge_index.shape[1]:5} edges")

# print(f"\n✓ Total: {sum(graph[nt].num_nodes for nt in graph.node_types)} nodes, "
#       f"{sum(graph[et].edge_index.shape[1] for et in graph.edge_types)} edges")

# if hasattr(graph['api'], 'api_names'):
#     print(f"\n✓ Sample APIs: {', '.join(graph['api'].api_names[:5])}")

# if hasattr(graph, 'graph_features'):
#     print(f"\n✓ Graph features: {graph.graph_features.shape}")


# ascii_chain.py
import pickle
from pathlib import Path
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("idx", type=int, nargs="?", default=0)
    ap.add_argument("--cache", type=str, default="../cache/train")
    ap.add_argument("--n", type=int, default=25)
    args = ap.parse_args()

    files = sorted(Path(args.cache).glob("*.pkl"))
    graph = pickle.load(open(files[args.idx], "rb"))

    api_names = getattr(graph["api"], "api_names", None)

    ei = graph[("api","temporal","api")].edge_index
    src = ei[0].tolist()
    dst = ei[1].tolist()
    next_map = {}
    for s,d in zip(src,dst):
        if s not in next_map:
            next_map[s]=d

    chain=[0]
    for _ in range(args.n-1):
        cur=chain[-1]
        if cur not in next_map: break
        chain.append(next_map[cur])

    def label(i):
        return api_names[i] if api_names and i < len(api_names) else f"api[{i}]"

    print("ASCII view (temporal chain):\n")
    for i, node in enumerate(chain):
        prefix = "└─" if i == len(chain)-1 else "├─"
        indent = "   " * i
        print(f"{indent}{prefix} {label(node)}")

if __name__ == "__main__":
    main()