# scripts/test_vocab.py

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from graph_construction.vocabulary import load_vocab, APIVocabulary

VOCAB_PATH = PROJECT_ROOT / "results" / "api_vocabulary.json"

# ── Test 1: Load the saved JSON ──────────────────────────────────
print("=== Test 1: Loading vocab from JSON ===")
vocab = load_vocab(VOCAB_PATH)          # ← correct: module-level function
assert vocab is not None, "load_vocab() returned None!"
print(f"  Vocab size  : {vocab.size()}")
print(f"  _next_idx   : {vocab._next_idx}")  # ← private attr in your class

# ── Test 2: Special tokens ───────────────────────────────────────
print("\n=== Test 2: Special tokens ===")
assert vocab.get_idx("<UNK>") == 0, "UNK not at index 0!"
assert vocab.get_idx("<PAD>") == 1, "PAD not at index 1!"
print(f"  <UNK> → {vocab.get_idx('<UNK>')}  ✅")
print(f"  <PAD> → {vocab.get_idx('<PAD>')}  ✅")

# ── Test 3: Known API roundtrip ──────────────────────────────────
print("\n=== Test 3: Known API roundtrip ===")
known_api = list(vocab.api_to_idx.keys())[2]   # first real API (index 2)
idx = vocab.get_idx(known_api)
back = vocab.get_api(idx)
assert back == known_api, f"Roundtrip failed: {known_api} → {idx} → {back}"
print(f"  '{known_api}' → {idx} → '{back}'  ✅")

# ── Test 4: Unknown API returns 0 ────────────────────────────────
print("\n=== Test 4: Unknown API ===")
assert vocab.get_idx("FakeAPI_XYZ_999") == 0
print(f"  'FakeAPI_XYZ_999' → {vocab.get_idx('FakeAPI_XYZ_999')}  ✅")

# ── Test 5: __contains__ works ───────────────────────────────────
print("\n=== Test 5: Membership test ===")
assert known_api in vocab
assert "FakeAPI_XYZ_999" not in vocab
print(f"  '{known_api}' in vocab → True  ✅")
print(f"  'FakeAPI_XYZ_999' in vocab → False  ✅")

# ── Test 6: Top 10 most frequent APIs ────────────────────────────
print("\n=== Test 6: Top 10 most frequent APIs ===")
for rank, (api, count) in enumerate(vocab.api_counts.most_common(10), 1):
    print(f"  {rank:>2}. [{vocab.get_idx(api):>4}]  {api:<40} ({count:,} calls)")

print("\n✅ All tests passed — vocabulary.py is working correctly!")
