"""
graphconstruction/vocabulary.py

API Vocabulary Builder for Heterogeneous GNN Malware Detection.

Scans the dataset (raw CAPE JSONs or cached PKL graphs) to build
a stable API name → integer index mapping used by the embedding layer.

Special tokens:
    <UNK>  = 0   (unseen API at inference time)
    <PAD>  = 1   (padding)

Saves / loads from:  results/api_vocabulary.json

Usage (standalone build):
    python -m graphconstruction.vocabulary

Usage (import):
    from graphconstruction.vocabulary import get_global_vocab, load_vocab
    vocab = get_global_vocab()          # returns singleton
    idx   = vocab.get_idx("NtCreateFile")
"""

import json
import logging
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

# ── project root on path ──────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).parent.parent))

from config import MAX_VOCAB_SIZE, CACHE_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
VOCAB_SAVE_PATH: Path = PROJECT_ROOT / "results" / "api_vocabulary.json"
ENRICHED_DIR: Path = Path(CACHE_DIR) / "enriched"


# ══════════════════════════════════════════════════════════════════════════════
#  APIVocabulary class
# ══════════════════════════════════════════════════════════════════════════════

class APIVocabulary:
    """
    Bidirectional API name ↔ index mapping with frequency tracking.

    Special tokens:
        index 0 → <UNK>  (unknown / unseen API)
        index 1 → <PAD>  (padding token)
    """

    SPECIAL_TOKENS = {"<UNK>": 0, "<PAD>": 1}

    def __init__(self, max_vocab_size: int = MAX_VOCAB_SIZE) -> None:
        self.max_vocab_size = max_vocab_size
        self.api_to_idx: Dict[str, int] = dict(self.SPECIAL_TOKENS)
        self.idx_to_api: Dict[int, str] = {v: k for k, v in self.SPECIAL_TOKENS.items()}
        self.api_counts: Counter = Counter()
        self._next_idx: int = len(self.SPECIAL_TOKENS)   # starts at 2

    # ── mutation ──────────────────────────────────────────────────────────────

    def add_api(self, api_name: str) -> None:
        """Register an API name; ignored if vocab is full or name is empty."""
        if not api_name or not isinstance(api_name, str):
            return
        self.api_counts[api_name] += 1
        if api_name not in self.api_to_idx and self._next_idx < self.max_vocab_size:
            self.api_to_idx[api_name] = self._next_idx
            self.idx_to_api[self._next_idx] = api_name
            self._next_idx += 1

    # ── lookup ────────────────────────────────────────────────────────────────

    def get_idx(self, api_name: str) -> int:
        """Return index for *api_name*; returns 0 (<UNK>) if not found."""
        return self.api_to_idx.get(api_name, 0)

    def get_api(self, idx: int) -> str:
        """Return API name for *idx*; returns '<UNK>' if not found."""
        return self.idx_to_api.get(idx, "<UNK>")

    # ── compatibility aliases (needed for Memory v2 helpers) ─────────────────

    def token_to_idx(self, token: str) -> int:
        """Alias for get_idx(...)."""
        return self.get_idx(token)

    def token_to_id(self, token: str) -> int:
        """Alias for get_idx(...)."""
        return self.get_idx(token)

    def idx_to_token(self, idx: int) -> str:
        """Alias for get_api(...)."""
        return self.get_api(idx)

    def get_count(self, api_name: str) -> int:
        """Return observed count for API/token name."""
        return int(self.api_counts.get(api_name, 0))

    def get_frequency_table(self) -> Dict[int, int]:
        """
        Return frequency table indexed by token id:
            {token_idx: count}
        """
        freq_table: Dict[int, int] = {}
        for api_name, count in self.api_counts.items():
            idx = self.get_idx(api_name)
            freq_table[idx] = int(count)
        return freq_table

    # ── properties ───────────────────────────────────────────────────────────

    def size(self) -> int:
        """Current vocabulary size (including special tokens)."""
        return len(self.api_to_idx)

    def __len__(self) -> int:
        return self.size()

    def __contains__(self, api_name: str) -> bool:
        return api_name in self.api_to_idx

    def __repr__(self) -> str:
        return (
            f"APIVocabulary(size={self.size()}, "
            f"max={self.max_vocab_size}, "
            f"unique_apis={self.size() - len(self.SPECIAL_TOKENS)})"
        )

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "api_to_idx": self.api_to_idx,
            "api_counts": dict(self.api_counts),
            "max_vocab_size": self.max_vocab_size,
            "vocab_size": self.size(),
        }

    @classmethod
    def from_dict(cls, data) -> "APIVocabulary":
        vocab = cls(max_vocab_size=data.get("max_vocab_size", MAX_VOCAB_SIZE))
        vocab.api_to_idx = {k: int(v) for k, v in data["api_to_idx"].items()}
        vocab.idx_to_api = {int(v): k for k, v in vocab.api_to_idx.items()}
        vocab.api_counts = Counter(data.get("api_counts", {}))
        vocab._next_idx = max(vocab.idx_to_api.keys(), default=1) + 1
        return vocab


# ══════════════════════════════════════════════════════════════════════════════
#  Save / Load helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_vocab(vocab: APIVocabulary, path: Path = VOCAB_SAVE_PATH) -> None:
    """Persist vocabulary to *path* as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info(f"Vocabulary saved → {path}  (size={vocab.size()})")


def load_vocab(path: Path = VOCAB_SAVE_PATH) -> Optional[APIVocabulary]:
    """
    Load vocabulary from *path*.
    Returns None if the file does not exist.
    """
    if not Path(path).exists():
        logger.warning(f"Vocabulary file not found: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    vocab = APIVocabulary.from_dict(data)
    logger.info(f"Vocabulary loaded ← {path}  (size={vocab.size()})")
    return vocab


# ══════════════════════════════════════════════════════════════════════════════
#  Singleton global vocabulary (used by graphbuilder.py)
# ══════════════════════════════════════════════════════════════════════════════

_GLOBAL_VOCAB: Optional[APIVocabulary] = None


def get_global_vocab() -> APIVocabulary:
    """
    Return the process-level singleton vocabulary.

    Priority:
        1. Already initialised in memory  → return as-is
        2. Saved JSON file exists         → load from disk
        3. Neither                        → return empty vocabulary
           (will be populated on first dataset pass)
    """
    global _GLOBAL_VOCAB
    if _GLOBAL_VOCAB is None:
        loaded = load_vocab()
        _GLOBAL_VOCAB = loaded if loaded is not None else APIVocabulary()
    return _GLOBAL_VOCAB


def reset_global_vocab() -> None:
    """Reset the singleton (useful during unit tests or vocabulary rebuilds)."""
    global _GLOBAL_VOCAB
    _GLOBAL_VOCAB = None


# ══════════════════════════════════════════════════════════════════════════════
#  Vocabulary builder  (scans enriched PKL graphs)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_api_names_from_graph(graph) -> List[str]:
    """
    Pull API names out of a cached HeteroData graph.

    Strategy (tries in order):
        1. graph['api'].api_names   — list of raw strings stored at build time
        2. graph['api'].names       — legacy attribute name
    Returns an empty list if neither attribute exists.
    """
    for attr in ("api_names", "names"):
        names = getattr(graph.get("api", object()), attr, None)
        if names:
            return list(names)
    return []


def build_vocab_from_enriched_graphs(
    enriched_dir: Path = ENRICHED_DIR,
    save_path: Path = VOCAB_SAVE_PATH,
    max_vocab_size: int = MAX_VOCAB_SIZE,
) -> APIVocabulary:
    """
    Scan every *.pkl graph in *enriched_dir* (train / val / test splits),
    collect all API names, and build a frequency-sorted vocabulary.

    The top-*max_vocab_size* most frequent APIs are kept; the rest map to
    <UNK> (index 0) at inference time.
    """
    enriched_dir = Path(enriched_dir)
    if not enriched_dir.exists():
        raise FileNotFoundError(f"Enriched graph directory not found: {enriched_dir}")

    splits = ["train", "val", "test"]
    all_pkl_files: List[Path] = []
    for split in splits:
        split_dir = enriched_dir / split
        if split_dir.exists():
            all_pkl_files.extend(sorted(split_dir.glob("*.pkl")))

    if not all_pkl_files:
        raise FileNotFoundError(f"No .pkl files found under {enriched_dir}")

    logger.info(f"Scanning {len(all_pkl_files):,} graph files for API vocabulary...")

    counter = Counter()

    for pkl_path in all_pkl_files:
        try:
            with open(pkl_path, "rb") as f:
                obj = pickle.load(f)

            if isinstance(obj, tuple):
                graph = obj[0]
            else:
                graph = obj

            names = _extract_api_names_from_graph(graph)
            counter.update([n for n in names if isinstance(n, str) and n])

        except Exception as e:
            logger.warning(f"Skipping {pkl_path.name}: {e}")
            continue

    vocab = APIVocabulary(max_vocab_size=max_vocab_size)

    # Keep special tokens, then add most common APIs
    most_common = counter.most_common(max(0, max_vocab_size - len(APIVocabulary.SPECIAL_TOKENS)))
    for api_name, count in most_common:
        vocab.api_counts[api_name] = count
        if api_name not in vocab.api_to_idx:
            vocab.api_to_idx[api_name] = vocab._next_idx
            vocab.idx_to_api[vocab._next_idx] = api_name
            vocab._next_idx += 1

    save_vocab(vocab, save_path)
    return vocab


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    vocab = build_vocab_from_enriched_graphs()
    print(vocab)