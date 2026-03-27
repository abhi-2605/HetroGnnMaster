from pathlib import Path
import random
import shutil
from config import ENRICHED_DIR

# ============================================================
# CONFIG
# ============================================================
# Folder containing ONLY the new v2 benign pkl files
NEW_V2_BENIGN_DIR = Path("./enrichedB/cuckoo_enriched_v2/all") 

# Existing split cache root
CACHE_ROOT = Path(ENRICHED_DIR)
TRAIN_DIR = CACHE_ROOT / "train"
VAL_DIR = CACHE_ROOT / "val"
TEST_DIR = CACHE_ROOT / "test"

TARGET_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

SEED = 42
COPY_FILES = True   # set False for dry run
PREFIX_TO_REMOVE = "ben_reports__"

# ============================================================
# HELPERS
# ============================================================
def strip_prefix(filename: str, prefix: str = PREFIX_TO_REMOVE) -> str:
    if filename.startswith(prefix):
        return filename[len(prefix):]
    return filename

def count_pkls(folder: Path) -> int:
    return sum(1 for _ in folder.glob("*.pkl"))

# ============================================================
# VALIDATION
# ============================================================
if not NEW_V2_BENIGN_DIR.exists():
    raise FileNotFoundError(f"NEW_V2_BENIGN_DIR not found: {NEW_V2_BENIGN_DIR}")

for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if not d.exists():
        raise FileNotFoundError(f"Missing split folder: {d}")

ratio_sum = sum(TARGET_RATIOS.values())
if abs(ratio_sum - 1.0) > 1e-9:
    raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum}")

new_files = sorted(NEW_V2_BENIGN_DIR.glob("*.pkl"))
if not new_files:
    raise ValueError(f"No .pkl files found in {NEW_V2_BENIGN_DIR}")

print("Before adding:")
print(f"  train: {count_pkls(TRAIN_DIR)}")
print(f"  val  : {count_pkls(VAL_DIR)}")
print(f"  test : {count_pkls(TEST_DIR)}")
print()

print(f"Found {len(new_files)} new v2 benign files")

# ============================================================
# SPLIT ASSIGNMENT
# ============================================================
random.seed(SEED)
random.shuffle(new_files)

n = len(new_files)
n_train = int(round(n * TARGET_RATIOS["train"]))
n_val = int(round(n * TARGET_RATIOS["val"]))
n_test = n - n_train - n_val

assignments = {
    "train": new_files[:n_train],
    "val": new_files[n_train:n_train + n_val],
    "test": new_files[n_train + n_val:],
}

print("Planned additions:")
for split_name in ["train", "val", "test"]:
    print(f"  {split_name}: {len(assignments[split_name])}")
print()

# ============================================================
# COLLISION CHECK AFTER RENAMING
# ============================================================
dest_dirs = {
    "train": TRAIN_DIR,
    "val": VAL_DIR,
    "test": TEST_DIR,
}

collisions = []
for split_name, files in assignments.items():
    dest_dir = dest_dirs[split_name]
    for src in files:
        new_name = strip_prefix(src.name)
        dst = dest_dir / new_name
        if dst.exists():
            collisions.append((split_name, src.name, new_name))

if collisions:
    print("Filename collisions found after prefix removal:")
    for split_name, old_name, new_name in collisions[:20]:
        print(f"  {split_name}: {old_name} -> {new_name}")
    raise RuntimeError(f"Aborting due to {len(collisions)} filename collisions")

# ============================================================
# COPY WITH RENAMING
# ============================================================
if COPY_FILES:
    for split_name, files in assignments.items():
        dest_dir = dest_dirs[split_name]
        for src in files:
            new_name = strip_prefix(src.name)
            dst = dest_dir / new_name
            shutil.copy2(src, dst)
    print("Copy complete.")
else:
    print("Dry run only. No files copied.")

print()
print("After adding:")
print(f"  train: {count_pkls(TRAIN_DIR)} (+{len(assignments['train'])})")
print(f"  val  : {count_pkls(VAL_DIR)} (+{len(assignments['val'])})")
print(f"  test : {count_pkls(TEST_DIR)} (+{len(assignments['test'])})")