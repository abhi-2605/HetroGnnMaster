"""
models/memory_utils.py

Utility helpers for TGNN memory.

This file keeps the existing utility role and adds Memory v2 helpers for:
- API/token index conversion
- memory regularization
- confidence-aware write scaling
- frequency-aware write scaling
- novelty-aware write scaling
- final selective write-strength composition
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Dict, Optional

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import config
from graph_construction.vocabulary import load_vocab


# ══════════════════════════════════════════════════════════════════════════════
# Existing helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_api_indices(api_names, vocab, device=None) -> torch.LongTensor:
    """
    Convert a list of API names to vocab indices.

    Unknown APIs map to <UNK>.
    """
    indices = [vocab.token_to_idx(name) for name in api_names]
    return torch.tensor(
        indices,
        dtype=torch.long,
        device=device if device is not None else "cpu",
    )


def memory_regularization_loss(memory_bank: torch.Tensor, lambda_reg: float = 1e-4) -> torch.Tensor:
    """
    Simple L2 regularization over memory parameters.
    """
    return lambda_reg * torch.norm(memory_bank, p=2)


# ══════════════════════════════════════════════════════════════════════════════
# Memory v2 helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_api_frequency_table(vocab_path: str | Path) -> Dict[int, int]:
    """
    Load token frequency counts from the vocabulary file and return
    {token_idx: count}.

    Assumes the vocabulary object exposes:
        - token_to_id / token_to_idx
        - api_counts or equivalent count mapping by token string

    If counts are missing, returns an empty dict.
    """
    vocab = load_vocab(str(vocab_path))

    freq_table: Dict[int, int] = {}

    api_counts = getattr(vocab, "api_counts", None)
    if api_counts is None:
        return freq_table

    for token, count in api_counts.items():
        try:
            idx = vocab.token_to_idx(token)
            freq_table[idx] = int(count)
        except Exception:
            continue

    return freq_table


def compute_prediction_confidence(logits: torch.Tensor) -> torch.Tensor:
    """
    Compute per-sample confidence as max softmax probability.

    Args:
        logits: [B, C]

    Returns:
        confidence: [B]
    """
    probs = F.softmax(logits, dim=-1)
    conf, _ = probs.max(dim=-1)
    return conf


def expand_graph_confidence_to_nodes(
    graph_confidence: torch.Tensor,
    batch_index: torch.Tensor,
) -> torch.Tensor:
    """
    Expand graph-level confidence [B] to node-level confidence [N]
    using the node batch index.

    Args:
        graph_confidence: [B]
        batch_index: [N] graph assignment per node

    Returns:
        node_confidence: [N]
    """
    return graph_confidence[batch_index]


def compute_confidence_write_factor(
    confidence: torch.Tensor,
    min_confidence: Optional[float] = None,
) -> torch.Tensor:
    """
    Convert confidence into a [0, 1] write factor.

    Behavior:
    - below threshold -> weak/no write
    - above threshold -> progressively stronger write

    Args:
        confidence: [N]
        min_confidence: optional override

    Returns:
        factor: [N] in [0, 1]
    """
    min_conf = (
        min_confidence
        if min_confidence is not None
        else getattr(config, "MEMORY_WRITE_MIN_CONFIDENCE", 0.60)
    )

    # Linear ramp above threshold
    factor = (confidence - min_conf) / max(1.0 - min_conf, 1e-6)
    factor = factor.clamp(min=0.0, max=1.0)
    return factor


def compute_frequency_write_factor(
    api_indices: torch.LongTensor,
    freq_table: Dict[int, int],
    alpha: Optional[float] = None,
    eps: Optional[float] = None,
    common_token_cap: Optional[float] = None,
) -> torch.Tensor:
    """
    Frequency-aware write scaling.

    Frequent tokens should not dominate memory simply because they appear
    everywhere. This is a SOFT penalty, not a hard suppression.

    Formula:
        factor_i = 1 / (freq_i + eps)^alpha
    then normalized to [0, 1] over the current batch
    and optionally lower-bounded by a cap for extremely common tokens.

    Args:
        api_indices: [N]
        freq_table: {token_idx: count}
        alpha: exponent controlling strength of penalty
        eps: numerical stability
        common_token_cap: lower cap for very common tokens

    Returns:
        factor: [N] in [cap, 1]
    """
    alpha = alpha if alpha is not None else getattr(config, "MEMORY_FREQ_ALPHA", 0.5)
    eps = eps if eps is not None else getattr(config, "MEMORY_FREQ_EPS", 1e-6)
    common_token_cap = (
        common_token_cap
        if common_token_cap is not None
        else getattr(config, "MEMORY_COMMON_TOKEN_WRITE_CAP", 0.25)
    )

    freqs = []
    for idx in api_indices.tolist():
        freqs.append(float(freq_table.get(int(idx), 1)))
    freq_tensor = torch.tensor(freqs, dtype=torch.float32, device=api_indices.device)

    raw = 1.0 / torch.pow(freq_tensor + eps, alpha)

    # Normalize within this batch to [0, 1]
    raw_min = raw.min()
    raw_max = raw.max()
    if float(raw_max - raw_min) < 1e-12:
        norm = torch.ones_like(raw)
    else:
        norm = (raw - raw_min) / (raw_max - raw_min)

    # keep a lower floor so common tokens are reduced, not erased
    norm = common_token_cap + (1.0 - common_token_cap) * norm
    return norm.clamp(min=0.0, max=1.0)


def compute_novelty_write_factor(
    write_candidate: torch.Tensor,
    memory_reference: torch.Tensor,
    temperature: Optional[float] = None,
    eps: Optional[float] = None,
) -> torch.Tensor:
    """
    Novelty-aware write scaling.

    If the current contextual write candidate is nearly identical to the
    retrieved/stored memory, the write should be weaker. If it is meaningfully
    different, the write can be stronger.

    Args:
        write_candidate: [N, D]
        memory_reference: [N, D]
        temperature: optional scaling
        eps: numerical stability

    Returns:
        factor: [N] in [0, 1]
    """
    temperature = (
        temperature
        if temperature is not None
        else getattr(config, "MEMORY_NOVELTY_TEMPERATURE", 1.0)
    )
    eps = eps if eps is not None else getattr(config, "MEMORY_NOVELTY_EPS", 1e-6)

    cand = F.normalize(write_candidate, dim=-1)
    ref = F.normalize(memory_reference, dim=-1)

    similarity = (cand * ref).sum(dim=-1).clamp(min=-1.0, max=1.0)   # [N]
    novelty = 1.0 - similarity                                        # higher = more novel

    novelty = novelty / max(temperature, eps)
    novelty = novelty.clamp(min=0.0, max=1.0)
    return novelty


def compose_write_strength(
    num_items: int,
    device: torch.device,
    base_strength: Optional[float] = None,
    confidence_factor: Optional[torch.Tensor] = None,
    frequency_factor: Optional[torch.Tensor] = None,
    novelty_factor: Optional[torch.Tensor] = None,
    epoch_factor: Optional[torch.Tensor] = None,
    min_strength: Optional[float] = None,
    max_strength: Optional[float] = None,
) -> torch.Tensor:
    """
    Compose the final selective write strength.

    Default composition:
        strength =
            base_strength
            * confidence_factor
            * frequency_factor
            * novelty_factor
            * epoch_factor

    Missing factors default to 1.

    Returns:
        strength: [N]
    """
    base_strength = (
        base_strength
        if base_strength is not None
        else getattr(config, "MEMORY_WRITE_BASE_STRENGTH", 1.0)
    )
    min_strength = (
        min_strength
        if min_strength is not None
        else getattr(config, "MEMORY_WRITE_MIN_STRENGTH", 0.05)
    )
    max_strength = (
        max_strength
        if max_strength is not None
        else getattr(config, "MEMORY_WRITE_MAX_STRENGTH", 1.0)
    )

    strength = torch.full(
        (num_items,),
        fill_value=float(base_strength),
        dtype=torch.float32,
        device=device,
    )

    if confidence_factor is not None:
        strength = strength * confidence_factor.to(device=device, dtype=torch.float32)

    if frequency_factor is not None:
        strength = strength * frequency_factor.to(device=device, dtype=torch.float32)

    if novelty_factor is not None:
        strength = strength * novelty_factor.to(device=device, dtype=torch.float32)

    if epoch_factor is not None:
        strength = strength * epoch_factor.to(device=device, dtype=torch.float32)

    return strength.clamp(min=min_strength, max=max_strength)


def compute_memory_epoch_factor(
    epoch: int,
    warmup_epochs: Optional[int] = None,
) -> float:
    """
    Delayed memory activation factor.

    Before warmup ends -> 0
    After warmup -> 1

    This keeps the behavior simple and stable.
    """
    warmup = (
        warmup_epochs
        if warmup_epochs is not None
        else getattr(config, "MEMORY_WARMUP_EPOCHS", 5)
    )
    return 0.0 if epoch < warmup else 1.0