"""Count the trained parameters of the reported model (thesis §5.1: ~16.1 million).

Run from the repository root:  python presentation/count_parameters.py
"""
import collections
import torch

CKPT = "results/checkpoints/best_model_1.pt"

state = torch.load(CKPT, map_location="cpu", weights_only=False)["model_state_dict"]
# learned weights only: skip batch-norm running statistics and counters
weights = {k: v for k, v in state.items()
           if v.is_floating_point() and "running_" not in k and "num_batches" not in k}

total = sum(v.numel() for v in weights.values())
print(f"Total trained parameters: {total:,}  (~{total / 1e6:.1f} million)\n")

parts = collections.Counter()
for name, v in weights.items():
    parts[name.split(".")[0]] += v.numel()
for name, n in parts.most_common():
    print(f"  {name:35s} {n:>12,}  {n / total:6.1%}")

# hand check on one small layer: a 128 -> 128 linear layer has 128*128 weights + 128 biases
for name, v in weights.items():
    if name.startswith("semantic_graph_projection") and name.endswith("weight") and v.dim() == 2:
        out_dim, in_dim = v.shape
        print(f"\nHand check, {name}: {in_dim} inputs x {out_dim} outputs = {in_dim * out_dim:,} weights"
              f" + {out_dim} biases = {in_dim * out_dim + out_dim:,} (reported above: {parts['semantic_graph_projection']:,})")
        break
