"""
models/memory_utils.py

Stateless helper utilities for SemanticMemory.

Functions:
    get_api_indices()              → convert API name list → LongTensor of indices
    aggregate_memory_by_graph()    → pool per-node memory → single graph vector
    memory_regularization_loss()   → L2 penalty to keep memory vectors stable
"""

from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from graph_construction.vocabulary import APIVocabulary


# ══════════════════════════════════════════════════════════════════════════════
#  1. get_api_indices
# ══════════════════════════════════════════════════════════════════════════════

def get_api_indices(
    api_names: list,
    vocab:     APIVocabulary,
    device:    torch.device = None,
) -> torch.LongTensor:
    """
    Convert a list of API name strings to a LongTensor of vocab indices.

    Unknown APIs → index 0  (<UNK>), never crashes.

    Args:
        api_names : list of str, e.g. ['NtClose', 'LdrLoadDll', ...]
        vocab     : loaded APIVocabulary instance
        device    : target torch device (optional)

    Returns:
        indices : LongTensor [N]

    Example:
        indices = get_api_indices(graph['api'].api_names, vocab)
        # tensor([4, 6, 2, 0, 19, ...])
    """
    indices = torch.tensor(
        [vocab.get_idx(name) for name in api_names],
        dtype=torch.long,
    )
    if device is not None:
        indices = indices.to(device)
    return indices


# ══════════════════════════════════════════════════════════════════════════════
#  2. aggregate_memory_by_graph
# ══════════════════════════════════════════════════════════════════════════════

def aggregate_memory_by_graph(
    memory_vectors: torch.Tensor,          # [N, 128]
    api_indices:    torch.LongTensor,      # [N]
    method:         str = "attention",     # "attention" | "mean" | "max"
) -> torch.Tensor:
    """
    Pool per-node memory vectors into a single graph-level vector.

    Three pooling strategies:

        "attention" (default)
            Weights each node by the L2-norm of its memory vector.
            Nodes with stronger/more-trained memories contribute more.
            Best for classification — focuses on discriminative APIs.

        "mean"
            Simple average across all nodes.
            Fast, stable, ignores importance.

        "max"
            Element-wise max across all nodes.
            Captures the peak signal per memory dimension.

    Args:
        memory_vectors : [N, 128]  per-node memory (output of SemanticMemory.read)
        api_indices    : [N]       API vocab indices (used to mask <PAD>)
        method         : pooling strategy

    Returns:
        graph_vector : [128]

    Example:
        mem_vectors  = memory(api_idx, node_feat, timestamps)[:, 775:]  # [N,128]
        graph_memory = aggregate_memory_by_graph(mem_vectors, api_idx)  # [128]
    """
    # Mask out <PAD> nodes (index 1) — they carry no information
    valid_mask = (api_indices != 1)                              # [N] bool
    if valid_mask.sum() == 0:
        return torch.zeros(memory_vectors.size(-1),
                           device=memory_vectors.device)

    valid_mem = memory_vectors[valid_mask]                       # [M, 128]

    if method == "mean":
        return valid_mem.mean(dim=0)                             # [128]

    elif method == "max":
        return valid_mem.max(dim=0).values                      # [128]

    else:  # "attention"
        # Weight by L2 norm of each memory vector
        norms   = valid_mem.norm(dim=-1, keepdim=True)          # [M, 1]
        weights = F.softmax(norms, dim=0)                       # [M, 1]
        return (weights * valid_mem).sum(dim=0)                 # [128]


# ══════════════════════════════════════════════════════════════════════════════
#  3. memory_regularization_loss
# ══════════════════════════════════════════════════════════════════════════════

def memory_regularization_loss(
    memory_bank_weight: torch.Tensor,    # [vocab_size, 128]
    lambda_reg:         float = 1e-4,
) -> torch.Tensor:
    """
    L2 regularisation penalty on the memory bank.

    Prevents memory vectors from growing unboundedly during long training.
    Add this as a small term to the total training loss.

    Loss = lambda_reg * mean(||memory_vector||^2)

    Typical lambda_reg values:
        1e-4  → gentle nudge (recommended for most runs)
        1e-3  → stronger constraint (if memory norms grow > 5.0)
        1e-5  → almost no constraint (if memory is underfitting)

    Args:
        memory_bank_weight : SemanticMemory.memory_bank.weight  [V, 128]
        lambda_reg         : regularisation strength

    Returns:
        scalar loss tensor (differentiable)

    Example:
        total_loss = task_loss + memory_regularization_loss(
            memory.memory_bank.weight
        )
    """
    # Skip special tokens <UNK>=0 and <PAD>=1
    real_memories = memory_bank_weight[2:]                       # [V-2, 128]
    l2_norms_sq   = real_memories.pow(2).sum(dim=-1)            # [V-2]
    return lambda_reg * l2_norms_sq.mean()


# ══════════════════════════════════════════════════════════════════════════════
#  Smoke test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from graph_construction.vocabulary import load_vocab
    from models.semantic_memory import build_semantic_memory

    print("=== memory_utils.py Smoke Test ===\n")

    vocab  = load_vocab("results/api_vocabulary.json")
    memory = build_semantic_memory()

    # ── Test 1: get_api_indices ───────────────────────────────────
    print("=== Test 1: get_api_indices ===")
    api_names = ["NtClose", "NtOpenKey", "FakeAPI_XYZ", "NtProtectVirtualMemory"]
    indices   = get_api_indices(api_names, vocab)
    print(f"  Input  : {api_names}")
    print(f"  Output : {indices.tolist()}")
    assert indices[2].item() == 0, "Unknown API should map to 0 (<UNK>)"
    assert indices.shape == (4,)
    print(f"  ✅ Unknown API → 0 (<UNK>), shape correct\n")

    # ── Test 2: aggregate_memory_by_graph ────────────────────────
    print("=== Test 2: aggregate_memory_by_graph ===")
    N          = 64
    api_idx    = torch.randint(2, 325, (N,))
    node_feat  = torch.randn(N, 775)
    timestamps = torch.linspace(0, 100, N)

    enriched   = memory(api_idx, node_feat, timestamps)          # [N, 903]
    mem_only   = enriched[:, 775:]                               # [N, 128]

    for method in ["attention", "mean", "max"]:
        g_vec = aggregate_memory_by_graph(mem_only, api_idx, method=method)
        assert g_vec.shape == (128,), f"Wrong shape for method={method}"
        print(f"  [{method:>9}] graph_vector shape: {list(g_vec.shape)}  ✅")

    # Test PAD masking
    api_idx_with_pad        = api_idx.clone()
    api_idx_with_pad[:10]   = 1                                  # first 10 = PAD
    g_vec_masked = aggregate_memory_by_graph(mem_only, api_idx_with_pad)
    assert g_vec_masked.shape == (128,)
    print(f"  [pad mask  ] PAD nodes correctly ignored  ✅\n")

    # ── Test 3: memory_regularization_loss ───────────────────────
    print("=== Test 3: memory_regularization_loss ===")
    reg_loss = memory_regularization_loss(memory.memory_bank.weight)
    assert reg_loss.shape == ()
    assert reg_loss.item() >= 0
    print(f"  reg_loss (λ=1e-4) : {reg_loss.item():.8f}  ✅")
    print(f"  Is differentiable : {reg_loss.requires_grad}  ✅\n")

    # ── Combined usage (how training loop will use them) ─────────
    print("=== Test 4: Combined training-loop pattern ===")
    task_loss  = torch.tensor(0.85, requires_grad=True)
    total_loss = task_loss + memory_regularization_loss(
        memory.memory_bank.weight
    )
    print(f"  task_loss  : {task_loss.item():.4f}")
    print(f"  total_loss : {total_loss.item():.8f}  ✅")
    print(f"  (reg adds  : {(total_loss - task_loss).item():.8f})\n")

    print("✅ memory_utils.py is working correctly!")
