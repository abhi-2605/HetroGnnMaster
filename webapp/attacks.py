"""
webapp/attacks.py

Behaviour-preserving attacks for the live demo.

Every attack here is the SAME implementation used for the robustness experiments
of the thesis; nothing is reimplemented. This module only gives them a uniform
signature, a display name, and an honest statement of what each one perturbs.

    graph, note = ATTACKS["reorder"].apply(graph, sample_id)

Each attack returns a NEW graph (the underlying functions deep-copy), so the
pristine analysed graph is never mutated.

On what "behaviour-preserving" means here: each transformation changes how the
execution was RECORDED, never what the program did. The malicious behaviour is
identical; only its surface form moves.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baseline_models import prove_classical_model_attacks as _classical  # noqa: E402
from baseline_models import compare_static_vs_existing_tgnn as _static  # noqa: E402
from scripts import test_semantic_substitution as _semantic  # noqa: E402


@dataclass(frozen=True)
class Attack:
    key: str
    label: str          # button text
    category: str       # "adversary" or "diagnostic" -- see note below
    targets: str        # which paradigm / branch this attack is aimed at
    description: str    # what it actually does to the trace
    _fn: Callable

    def apply(self, graph, sample_id: str) -> Tuple[object, Optional[float]]:
        out = self._fn(graph, sample_id)
        # the semantic substitutions return (graph, coverage_fraction)
        if isinstance(out, tuple):
            g, coverage = out[0], out[1]
            return g, (round(float(coverage), 4) if isinstance(coverage, (int, float)) else None)
        return out, None


# Two kinds of transformation, and the distinction is not cosmetic.
#
#   "adversary"  -- something a real attacker can actually do to a sample before
#                   it is recorded: rename its calls, reorder them, pad them,
#                   perturb the observed graph. The framework is expected to hold.
#
#   "diagnostic" -- an intervention on the model's own input representation, which
#                   no attacker has access to. Substituting embedding vectors is a
#                   probe of what the semantic branch relies on (the §5.5 design),
#                   not an attack anyone can mount. It is expected to hurt: that is
#                   what makes it evidence.
#
# Keeping them apart stops a diagnostic being read as a defeated attack.
ATTACKS: Dict[str, Attack] = {
    "rename": Attack(
        key="rename",
        category="adversary",
        label="Rename calls",
        targets="token identity",
        description=(
            "Renames a large fraction of API calls to meaning-preserving aliases and pads "
            "the trace with benign-looking activity. This is what a recompile or repack "
            "produces. It rewrites the names a counting model reads; the pretrained "
            "embeddings already attached to each node are left as they are."
        ),
        _fn=_classical.apply_frequency_dilution_attack,
    ),
    "rename_out_of_context": Attack(
        key="rename_out_of_context",
        category="adversary",
        label="Disguise calls",
        targets="token identity, harder",
        description=(
            "Replaces call names with real but behaviourally unrelated Windows APIs, the "
            "mimicry tactic. A model that re-encodes the name lands on the wrong "
            "behaviour rather than a near-synonym."
        ),
        _fn=_classical.apply_out_of_context_rename_attack,
    ),
    "reorder": Attack(
        key="reorder",
        category="adversary",
        label="Shuffle order",
        targets="temporal branch",
        description=(
            "Permutes the recorded order of the calls completely, so any detector keyed to "
            "a fixed call sequence loses its signal. The multiset of operations is untouched."
        ),
        _fn=_classical.apply_full_shuffle_attack,
    ),
    "interleave": Attack(
        key="interleave",
        category="adversary",
        label="Interleave benign calls",
        targets="adjacent-order statistics",
        description=(
            "Threads benign calls through the malicious sequence, diluting the local "
            "ordering patterns that bigram-style features depend on."
        ),
        _fn=_classical.apply_benign_interleaving_attack,
    ),
    "structure": Attack(
        key="structure",
        category="adversary",
        label="Perturb graph",
        targets="structural branch",
        description=(
            "Drops and rewires edges and injects decoy nodes, as an incompletely observed "
            "or deliberately noised execution would appear. The recorded API activity is "
            "left intact."
        ),
        _fn=_static.apply_structure_attack_local,
    ),
    "semantic_swap": Attack(
        key="semantic_swap",
        category="diagnostic",
        label="Swap meanings (synonym)",
        targets="semantic branch",
        description=(
            "Replaces each API embedding with that of its nearest functional neighbour. "
            "Identity changes, meaning is preserved. This is the §5.5 meaning-preserving track."
        ),
        _fn=_semantic.apply_semantic_vector_substitution,
    ),
    "semantic_random": Attack(
        key="semantic_random",
        category="diagnostic",
        label="Destroy meanings",
        targets="semantic branch, control",
        description=(
            "Overwrites each API embedding with an unrelated random vector, destroying "
            "meaning rather than exchanging it. The §5.5 control against which the "
            "synonym track is read."
        ),
        _fn=_semantic.apply_random_vector_substitution,
    ),
}


def list_attacks() -> list[dict]:
    """Serialisable catalogue for the UI."""
    return [
        {
            "key": a.key,
            "label": a.label,
            "category": a.category,
            "targets": a.targets,
            "description": a.description,
        }
        for a in ATTACKS.values()
    ]


def get(key: str) -> Attack:
    if key not in ATTACKS:
        raise KeyError(f"Unknown attack '{key}'. Available: {', '.join(ATTACKS)}")
    return ATTACKS[key]
