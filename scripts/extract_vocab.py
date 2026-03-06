# scripts/extract_vocab.py

import pickle
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import config
from graph_construction.vocabulary import APIVocabulary

ENRICHED_DIR = Path(config.ENRICHED_DIR)
VOCAB_SAVE_PATH = Path(getattr(config, "VOCAB_SAVE_PATH", "results/api_vocabulary.json"))


def extract_and_save_vocab():
    vocab = APIVocabulary(max_vocab_size=config.MAX_VOCAB_SIZE)

    pkl_files = [
        p for split in ["train", "val", "test"]
        for p in (ENRICHED_DIR / split).glob("*.pkl")
    ]

    print(f"[extract_vocab] Scanning {len(pkl_files)} graphs...")

    for pkl_path in pkl_files:
        try:
            with open(pkl_path, "rb") as f:
                graph = pickle.load(f)
            for api_name in graph["api"].api_names:
                vocab.add_api(api_name)
        except Exception as e:
            print(f"  [WARN] Skipping {pkl_path.name}: {e}")

    print(f"[extract_vocab] Done. Vocab size: {vocab.size()} unique API tokens.")

    # Save manually — no dependency on vocab.save() in vocabulary.py
    VOCAB_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "max_vocab_size": vocab.max_vocab_size,
        "api_to_idx": vocab.api_to_idx,
        "api_counts": dict(vocab.api_counts),
    }
    with open(VOCAB_SAVE_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[Vocabulary] Saved {vocab.size()} tokens → {VOCAB_SAVE_PATH}")


if __name__ == "__main__":
    extract_and_save_vocab()
