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

from config import MAX_VOCAB_SIZE, CACHE_DIR, PROJECT_ROOT   # adjust names to match your config.py

logger = logging.getLogger(__name__)

# ── paths ─────────────────────────────────────────────────────────────────────
VOCAB_SAVE_PATH: Path = PROJECT_ROOT / "results" / "api_vocabulary.json"
ENRICHED_DIR:    Path = Path(CACHE_DIR) / "enriched"


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
            "api_to_idx":     self.api_to_idx,
            "api_counts":     dict(self.api_counts),
            "max_vocab_size": self.max_vocab_size,
            "vocab_size":     self.size(),
        }

    @classmethod
    def from_dict(cls, data) -> "APIVocabulary":
        vocab = cls(max_vocab_size=data.get("max_vocab_size", MAX_VOCAB_SIZE))
        vocab.api_to_idx  = {k: int(v) for k, v in data["api_to_idx"].items()}
        vocab.idx_to_api  = {int(v): k for k, v in vocab.api_to_idx.items()}
        vocab.api_counts  = Counter(data.get("api_counts", {}))
        vocab._next_idx   = max(vocab.idx_to_api.keys(), default=1) + 1
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
        raise FileNotFoundError(f"No *.pkl files found under {enriched_dir}")

    logger.info(f"Scanning {len(all_pkl_files)} graphs in {enriched_dir} …")

    counter: Counter = Counter()
    errors = 0

    for pkl_path in all_pkl_files:
        try:
            with open(pkl_path, "rb") as f:
                graph = pickle.load(f)
            names = _extract_api_names_from_graph(graph)
            counter.update(names)
        except Exception as e:
            logger.debug(f"  skip {pkl_path.name}: {e}")
            errors += 1

    if errors:
        logger.warning(f"  {errors} files could not be read (see DEBUG log).")

    # ── build ordered vocabulary (most-frequent first) ────────────────────────
    vocab = APIVocabulary(max_vocab_size=max_vocab_size)
    for api_name, _ in counter.most_common(max_vocab_size - len(APIVocabulary.SPECIAL_TOKENS)):
        vocab.add_api(api_name)
    # restore full counts
    vocab.api_counts = counter

    logger.info(
        f"Vocabulary built: {vocab.size()} entries "
        f"({vocab.size() - len(APIVocabulary.SPECIAL_TOKENS)} unique APIs) "
        f"from {len(all_pkl_files) - errors} graphs."
    )

    save_vocab(vocab, save_path)
    return vocab


def build_vocab_from_cape_jsons(
    raw_data_dir: Path,
    save_path: Path = VOCAB_SAVE_PATH,
    max_vocab_size: int = MAX_VOCAB_SIZE,
) -> APIVocabulary:
    """
    Alternative builder: scan raw CAPE JSON report files directly.

    Looks for the ``"behavior" → "processes" → "calls" → "api"`` path
    inside each JSON report.

    Args:
        raw_data_dir: Root folder that contains *.json CAPE report files
                      (searched recursively).
    """
    import glob

    raw_data_dir = Path(raw_data_dir)
    json_files = list(raw_data_dir.rglob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found under {raw_data_dir}")

    logger.info(f"Scanning {len(json_files)} CAPE JSON reports …")

    counter: Counter = Counter()
    errors = 0

    for jpath in json_files:
        try:
            with open(jpath, "r", encoding="utf-8", errors="replace") as f:
                report = json.load(f)
            processes = (
                report.get("behavior", {})
                      .get("processes", [])
            )
            for proc in processes:
                for call in proc.get("calls", []):
                    api_name = call.get("api", "")
                    if api_name:
                        counter[api_name] += 1
        except Exception as e:
            logger.debug(f"  skip {jpath.name}: {e}")
            errors += 1

    if errors:
        logger.warning(f"  {errors} files could not be read.")

    vocab = APIVocabulary(max_vocab_size=max_vocab_size)
    for api_name, _ in counter.most_common(max_vocab_size - len(APIVocabulary.SPECIAL_TOKENS)):
        vocab.add_api(api_name)
    vocab.api_counts = counter

    logger.info(f"Vocabulary built: {vocab.size()} entries from JSON reports.")
    save_vocab(vocab, save_path)
    return vocab


# ══════════════════════════════════════════════════════════════════════════════
#  CLI  (python -m graphconstruction.vocabulary)
# ══════════════════════════════════════════════════════════════════════════════

def _cli() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Build or inspect the API vocabulary for the GNN pipeline."
    )
    parser.add_argument(
        "--source", choices=["graphs", "jsons"], default="graphs",
        help="'graphs' = scan enriched PKL files (default); "
             "'jsons'  = scan raw CAPE JSON reports.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Path to raw CAPE JSON directory (only needed for --source jsons).",
    )
    parser.add_argument(
        "--enriched-dir", type=Path, default=ENRICHED_DIR,
        help=f"Enriched graph directory  [default: {ENRICHED_DIR}]",
    )
    parser.add_argument(
        "--save-path", type=Path, default=VOCAB_SAVE_PATH,
        help=f"Output JSON path  [default: {VOCAB_SAVE_PATH}]",
    )
    parser.add_argument(
        "--max-vocab", type=int, default=MAX_VOCAB_SIZE,
        help=f"Maximum vocabulary size  [default: {MAX_VOCAB_SIZE}]",
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Print summary of an already-saved vocabulary and exit.",
    )
    args = parser.parse_args()

    # ── inspect mode ──────────────────────────────────────────────────────────
    if args.inspect:
        vocab = load_vocab(args.save_path)
        if vocab is None:
            print(f"No vocabulary found at {args.save_path}")
            sys.exit(1)
        print(f"\n{'='*60}")
        print(f"  Vocabulary: {args.save_path}")
        print(f"  Total entries  : {vocab.size()}")
        print(f"  Unique APIs    : {vocab.size() - len(APIVocabulary.SPECIAL_TOKENS)}")
        print(f"  Max vocab size : {vocab.max_vocab_size}")
        print(f"\n  Top-20 most frequent APIs:")
        for rank, (api, cnt) in enumerate(vocab.api_counts.most_common(20), 1):
            idx = vocab.get_idx(api)
            print(f"    {rank:>3}. [{idx:>5}]  {api:<45} ({cnt:,} calls)")
        print(f"{'='*60}\n")
        return

    # ── build mode ────────────────────────────────────────────────────────────
    if args.source == "graphs":
        vocab = build_vocab_from_enriched_graphs(
            enriched_dir=args.enriched_dir,
            save_path=args.save_path,
            max_vocab_size=args.max_vocab,
        )
    else:
        if args.data_dir is None:
            parser.error("--data-dir is required when --source jsons")
        vocab = build_vocab_from_cape_jsons(
            raw_data_dir=args.data_dir,
            save_path=args.save_path,
            max_vocab_size=args.max_vocab,
        )

    print(f"\n✅  {vocab}")
    print(f"    Saved to: {args.save_path}\n")


if __name__ == "__main__":
    _cli()
