# #!/usr/bin/env python3
# """
# Idempotent zero-day/train swap script.

# Behavior:
# 1) Move files from cache/zero_day_unseen/all -> cache/enriched/train
# 2) Remove prefix like mal_reports__ from filenames
# 3) Randomly move 145 malware files from train -> cache/zero_day_unseen/old
# 4) Update metadata CSV
# 5) Keep a state file so reruns do not repeat the swap

# Safe rerun behavior:
# - If swap already completed, it exits without doing anything.
# - If files were already moved before a previous crash, it reconciles state.
# """

# from __future__ import annotations

# import csv
# import json
# import random
# import re
# import shutil
# from pathlib import Path
# from typing import Dict, List, Optional, Tuple

# import pandas as pd

# # ============================================================
# # CONFIG
# # ============================================================

# RANDOM_SEED = 42

# PROJECT_ROOT = Path(__file__).resolve().parent.parent

# CACHE_DIR = PROJECT_ROOT / "cache"
# ENRICHED_DIR = CACHE_DIR / "enriched"
# TRAIN_DIR = ENRICHED_DIR / "train"
# VAL_DIR = ENRICHED_DIR / "val"
# TEST_DIR = ENRICHED_DIR / "test"

# ZERO_DAY_DIR = CACHE_DIR / "zero_day_unseen"
# ZERO_DAY_ALL_DIR = ZERO_DAY_DIR / "all"
# ZERO_DAY_OLD_DIR = ZERO_DAY_DIR / "old"

# ROOT_METADATA_PATH = PROJECT_ROOT / "extracted_metadata_appended.csv"
# DATA_METADATA_PATH = PROJECT_ROOT / "data" / "extracted_metadata_appended.csv"
# METADATA_PATH = ROOT_METADATA_PATH if ROOT_METADATA_PATH.exists() else DATA_METADATA_PATH

# UPDATED_METADATA_PATH = METADATA_PATH.with_name(METADATA_PATH.stem + "_updated.csv")
# MOVEMENT_LOG_PATH = PROJECT_ROOT / "swap_movement_log.csv"
# STATE_PATH = PROJECT_ROOT / "zero_day_swap_state.json"

# REMOVE_RANDOM_MALWARE_COUNT = 145

# PREFIX_PATTERNS = [
#     r"^mal_reports__+",
#     r"^mal_reports_+",
# ]

# STRICT_ON_FILENAME_CONFLICT = True
# OVERWRITE_ORIGINAL_METADATA = False

# # If metadata row for incoming sample is missing, append placeholder row
# APPEND_MISSING_METADATA_ROWS = True

# # ============================================================
# # HELPERS
# # ============================================================

# def set_seed(seed: int = RANDOM_SEED) -> None:
#     random.seed(seed)

# def ensure_dir(path: Path) -> None:
#     path.mkdir(parents=True, exist_ok=True)

# def strip_known_prefixes(filename: str) -> str:
#     new_name = filename
#     for pattern in PREFIX_PATTERNS:
#         new_name = re.sub(pattern, "", new_name)
#     return new_name

# def sample_id_from_name(filename: str) -> str:
#     clean = strip_known_prefixes(filename)
#     return Path(clean).stem

# def is_benign(category_name: Optional[str]) -> bool:
#     if category_name is None or pd.isna(category_name):
#         return False
#     return str(category_name).strip().lower() == "benign"

# def is_ransomware(category_name: Optional[str]) -> bool:
#     if category_name is None or pd.isna(category_name):
#         return False
#     return str(category_name).strip().lower() == "ransomware"

# def is_other_malware(category_name: Optional[str]) -> bool:
#     if category_name is None or pd.isna(category_name):
#         return False
#     label = str(category_name).strip().lower()
#     return label not in {"benign", "ransomware"}

# def build_file_index(folder: Path) -> Dict[str, Path]:
#     index = {}
#     if not folder.exists():
#         return index
#     for p in folder.iterdir():
#         if p.is_file():
#             index[sample_id_from_name(p.name)] = p
#     return index

# def infer_split_column(df: pd.DataFrame) -> pd.DataFrame:
#     if "split" in df.columns:
#         return df

#     train_ids = set(build_file_index(TRAIN_DIR).keys())
#     val_ids = set(build_file_index(VAL_DIR).keys())
#     test_ids = set(build_file_index(TEST_DIR).keys())
#     zero_all_ids = set(build_file_index(ZERO_DAY_ALL_DIR).keys())
#     zero_old_ids = set(build_file_index(ZERO_DAY_OLD_DIR).keys())

#     splits = []
#     for _, row in df.iterrows():
#         md5 = str(row["md5"]).strip() if "md5" in df.columns and pd.notna(row["md5"]) else None
#         filename = str(row["filename"]).strip() if "filename" in df.columns and pd.notna(row["filename"]) else None

#         candidates = []
#         if md5:
#             candidates.append(md5)
#         if filename:
#             candidates.append(sample_id_from_name(filename))

#         split = "unknown"
#         for sid in candidates:
#             if sid in train_ids:
#                 split = "train"
#                 break
#             if sid in val_ids:
#                 split = "val"
#                 break
#             if sid in test_ids:
#                 split = "test"
#                 break
#             if sid in zero_all_ids:
#                 split = "zero_day_all"
#                 break
#             if sid in zero_old_ids:
#                 split = "zero_day_old"
#                 break

#         splits.append(split)

#     df = df.copy()
#     df["split"] = splits
#     return df

# def build_metadata_index(df: pd.DataFrame) -> Dict[str, int]:
#     index = {}
#     for i, row in df.iterrows():
#         md5 = str(row["md5"]).strip() if "md5" in df.columns and pd.notna(row["md5"]) else None
#         filename = str(row["filename"]).strip() if "filename" in df.columns and pd.notna(row["filename"]) else None

#         sid = md5 or (sample_id_from_name(filename) if filename else None)
#         if sid and sid not in index:
#             index[sid] = i
#     return index

# def find_train_malware_candidates(df: pd.DataFrame, train_file_ids: set[str]) -> List[Tuple[str, int]]:
#     candidates = []
#     for row_idx, row in df.iterrows():
#         category_name = row.get("category_name", None)
#         split = row.get("split", "unknown")

#         md5 = str(row["md5"]).strip() if pd.notna(row.get("md5")) else None
#         filename = str(row["filename"]).strip() if pd.notna(row.get("filename")) else None

#         sid = md5 or (sample_id_from_name(filename) if filename else None)
#         if not sid:
#             continue

#         if split == "train" and sid in train_file_ids and is_other_malware(category_name):
#             candidates.append((sid, row_idx))
#     return candidates

# def safe_move(src: Path, dst: Path) -> None:
#     ensure_dir(dst.parent)
#     shutil.move(str(src), str(dst))

# def rename_if_needed(src: Path, cleaned_name: str) -> Path:
#     if src.name == cleaned_name:
#         return src

#     dst = src.with_name(cleaned_name)
#     if dst.exists() and dst.resolve() != src.resolve():
#         raise FileExistsError(f"Cannot rename {src.name} -> {cleaned_name}; target already exists: {dst}")
#     src.rename(dst)
#     return dst

# def get_nonconflicting_old_path(dst: Path) -> Path:
#     """
#     If target exists in zero_day_unseen/old, create a non-conflicting name:
#       file.pkl -> file__dup1.pkl -> file__dup2.pkl ...
#     """
#     if not dst.exists():
#         return dst

#     stem = dst.stem
#     suffix = dst.suffix
#     parent = dst.parent
#     i = 1
#     while True:
#         candidate = parent / f"{stem}__dup{i}{suffix}"
#         if not candidate.exists():
#             return candidate
#         i += 1

# def ensure_audit_columns(df: pd.DataFrame) -> pd.DataFrame:
#     df = df.copy()
#     for col in ["last_swap_action", "last_swap_note"]:
#         if col not in df.columns:
#             df[col] = ""
#     return df

# def append_placeholder_metadata_row(df: pd.DataFrame, sample_id: str, filename: str, split: str) -> pd.DataFrame:
#     row = {col: "" for col in df.columns}

#     if "md5" in df.columns:
#         row["md5"] = sample_id
#     if "filename" in df.columns:
#         row["filename"] = filename
#     if "split" in df.columns:
#         row["split"] = split
#     if "category_name" in df.columns:
#         row["category_name"] = "other"
#     if "last_swap_action" in df.columns:
#         row["last_swap_action"] = "placeholder_metadata_added"
#     if "last_swap_note" in df.columns:
#         row["last_swap_note"] = "Row appended because sample missing from metadata CSV"

#     return pd.concat([df, pd.DataFrame([row])], ignore_index=True)

# def load_state() -> dict:
#     if STATE_PATH.exists():
#         with open(STATE_PATH, "r", encoding="utf-8") as f:
#             return json.load(f)
#     return {"completed": False, "moved_in": [], "moved_out": []}

# def save_state(state: dict) -> None:
#     with open(STATE_PATH, "w", encoding="utf-8") as f:
#         json.dump(state, f, indent=2)

# # ============================================================
# # MAIN
# # ============================================================

# def main() -> None:
#     set_seed(RANDOM_SEED)

#     for needed in [ZERO_DAY_ALL_DIR, TRAIN_DIR]:
#         if not needed.exists():
#             raise FileNotFoundError(f"Required folder does not exist: {needed}")

#     ensure_dir(ZERO_DAY_OLD_DIR)

#     if not METADATA_PATH.exists():
#         raise FileNotFoundError(f"Metadata CSV not found: {METADATA_PATH}")

#     state = load_state()
#     if state.get("completed"):
#         print("[INFO] Swap already completed earlier. Exiting without changes.")
#         print(f"[INFO] State file: {STATE_PATH}")
#         return

#     df = pd.read_csv(METADATA_PATH)
#     df = infer_split_column(df)
#     df = ensure_audit_columns(df)

#     metadata_index = build_metadata_index(df)
#     movement_rows = []

#     # ------------------------------------------------
#     # STEP 1: move new unseen files into train
#     # ------------------------------------------------
#     all_new_files = sorted([p for p in ZERO_DAY_ALL_DIR.iterdir() if p.is_file()])
#     print(f"[INFO] Found {len(all_new_files)} files currently in zero_day_unseen/all")

#     moved_in_sample_ids = []
#     for src in all_new_files:
#         original_name = src.name
#         cleaned_name = strip_known_prefixes(original_name)
#         normalized_src = rename_if_needed(src, cleaned_name)

#         sid = sample_id_from_name(cleaned_name)
#         dst = TRAIN_DIR / cleaned_name

#         if dst.exists():
#             existing_sid = sample_id_from_name(dst.name)
#             if existing_sid != sid:
#                 msg = f"Target conflict in train: {dst}"
#                 if STRICT_ON_FILENAME_CONFLICT:
#                     raise FileExistsError(msg)
#                 print(f"[WARN] {msg}; skipping.")
#                 continue

#             print(f"[INFO] Already present in train, treating as already moved: {cleaned_name}")
#         else:
#             safe_move(normalized_src, dst)
#             print(f"[INFO] Moved into train: {cleaned_name}")

#         moved_in_sample_ids.append(sid)

#         row_idx = metadata_index.get(sid)
#         if row_idx is not None:
#             df.at[row_idx, "filename"] = cleaned_name
#             df.at[row_idx, "split"] = "train"
#             df.at[row_idx, "last_swap_action"] = "zero_day_all_to_train"
#             df.at[row_idx, "last_swap_note"] = f"Renamed from {original_name} -> {cleaned_name}"
#         else:
#             print(f"[WARN] No metadata row found for moved-in sample: {cleaned_name} (sample_id={sid})")
#             if APPEND_MISSING_METADATA_ROWS:
#                 df = append_placeholder_metadata_row(df, sid, cleaned_name, "train")
#                 metadata_index = build_metadata_index(df)

#         movement_rows.append({
#             "sample_id": sid,
#             "action": "zero_day_all_to_train",
#             "src": str(ZERO_DAY_ALL_DIR / original_name),
#             "dst": str(dst),
#             "old_filename": original_name,
#             "new_filename": cleaned_name,
#         })

#     print(f"[INFO] Total moved/recognized into train: {len(moved_in_sample_ids)}")

#     # ------------------------------------------------
#     # STEP 2: move random malware from train to old
#     # ------------------------------------------------
#     train_index = build_file_index(TRAIN_DIR)
#     train_file_ids = set(train_index.keys())

#     malware_candidates = find_train_malware_candidates(df, train_file_ids)
#     print(f"[INFO] Found {len(malware_candidates)} malware candidates currently in train")

#     if len(malware_candidates) < REMOVE_RANDOM_MALWARE_COUNT:
#         raise RuntimeError(
#             f"Not enough train malware samples to remove. "
#             f"Need {REMOVE_RANDOM_MALWARE_COUNT}, found {len(malware_candidates)}"
#         )

#     selected_to_remove = random.sample(malware_candidates, REMOVE_RANDOM_MALWARE_COUNT)

#     removed_sample_ids = []
#     for sid, row_idx in selected_to_remove:
#         src = train_index.get(sid)
#         if src is None or not src.exists():
#             print(f"[WARN] Train file missing for selected sample_id={sid}; skipping")
#             continue

#         dst = get_nonconflicting_old_path(ZERO_DAY_OLD_DIR / src.name)
#         safe_move(src, dst)
#         removed_sample_ids.append(sid)

#         df.at[row_idx, "split"] = "zero_day_old"
#         df.at[row_idx, "last_swap_action"] = "train_to_zero_day_old"
#         note = "Moved random malware out of train to old set"
#         if dst.name != src.name:
#             note += f"; renamed on move to avoid conflict: {src.name} -> {dst.name}"
#         df.at[row_idx, "last_swap_note"] = note

#         movement_rows.append({
#             "sample_id": sid,
#             "action": "train_to_zero_day_old",
#             "src": str(src),
#             "dst": str(dst),
#             "old_filename": src.name,
#             "new_filename": dst.name,
#         })

#     print(f"[INFO] Removed {len(removed_sample_ids)} malware samples from train to zero_day_unseen/old")

#     # ------------------------------------------------
#     # STEP 3: write CSV, log, state
#     # ------------------------------------------------
#     df.to_csv(UPDATED_METADATA_PATH, index=False)

#     with open(MOVEMENT_LOG_PATH, "w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(
#             f,
#             fieldnames=["sample_id", "action", "src", "dst", "old_filename", "new_filename"]
#         )
#         writer.writeheader()
#         writer.writerows(movement_rows)

#     if OVERWRITE_ORIGINAL_METADATA:
#         shutil.copy2(UPDATED_METADATA_PATH, METADATA_PATH)
#         print(f"[INFO] Original metadata overwritten: {METADATA_PATH}")

#     state["completed"] = True
#     state["moved_in"] = moved_in_sample_ids
#     state["moved_out"] = removed_sample_ids
#     save_state(state)

#     print(f"[INFO] Updated metadata written to: {UPDATED_METADATA_PATH}")
#     print(f"[INFO] Movement log written to: {MOVEMENT_LOG_PATH}")
#     print(f"[INFO] State file written to: {STATE_PATH}")

#     print("\n===== SUMMARY =====")
#     print(f"New files moved into train:      {len(moved_in_sample_ids)}")
#     print(f"Random malware moved to old:     {len(removed_sample_ids)}")
#     print(f"Random seed used:                {RANDOM_SEED}")
#     print(f"Updated CSV path:                {UPDATED_METADATA_PATH}")
#     print(f"Movement log path:               {MOVEMENT_LOG_PATH}")
#     print(f"State path:                      {STATE_PATH}")

# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
"""
Find PKL files in cache/enriched/{train,val,test} that are missing from
extracted_metadata_appended.csv, append new rows for them, and print the new md5 list.

Rules for appended rows:
- filename        -> actual PKL filename
- md5             -> filename stem
- all_categories  -> malware

Everything else is left blank unless the column exists and we can fill something sensible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CACHE_DIR = PROJECT_ROOT / "cache"
ENRICHED_DIR = CACHE_DIR / "enriched"

TRAIN_DIR = ENRICHED_DIR / "train"
VAL_DIR = ENRICHED_DIR / "val"
TEST_DIR = ENRICHED_DIR / "test"

ROOT_METADATA_PATH = PROJECT_ROOT / "extracted_metadata_appended.csv"
DATA_METADATA_PATH = PROJECT_ROOT / "data" / "extracted_metadata_appended.csv"
METADATA_PATH = ROOT_METADATA_PATH if ROOT_METADATA_PATH.exists() else DATA_METADATA_PATH

OUTPUT_METADATA_PATH = METADATA_PATH.with_name(METADATA_PATH.stem + "_with_missing_pkls_added.csv")


# ============================================================
# HELPERS
# ============================================================

def collect_pkls_with_split(folder: Path, split_name: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not folder.exists():
        return rows

    for p in sorted(folder.glob("*.pkl")):
        rows.append(
            {
                "filename": p.name,
                "md5": p.stem,
                "split": split_name,
                "filepath": str(p),
            }
        )
    return rows


def build_new_row(df: pd.DataFrame, pkl_info: Dict[str, str]) -> Dict[str, object]:
    """
    Build a row matching the existing CSV schema as much as possible.
    Only fill requested fields + a few safe helpers if those columns exist.
    """
    row = {col: "" for col in df.columns}

    if "filename" in df.columns:
        row["filename"] = pkl_info["filename"]

    if "md5" in df.columns:
        row["md5"] = pkl_info["md5"]

    if "all_categories" in df.columns:
        row["all_categories"] = "malware"

    # Optional helpful fields if present in your schema
    if "category_name" in df.columns:
        row["category_name"] = "malware"

    if "all_category_names" in df.columns:
        row["all_category_names"] = "malware"

    if "split" in df.columns:
        row["split"] = pkl_info["split"]

    if "filepath" in df.columns:
        row["filepath"] = pkl_info["filepath"]

    return row


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_PATH}")

    df = pd.read_csv(METADATA_PATH)

    if "md5" not in df.columns:
        raise ValueError("CSV must contain an 'md5' column.")

    # Collect all PKLs from train/val/test
    pkl_rows: List[Dict[str, str]] = []
    pkl_rows.extend(collect_pkls_with_split(TRAIN_DIR, "train"))
    pkl_rows.extend(collect_pkls_with_split(VAL_DIR, "val"))
    pkl_rows.extend(collect_pkls_with_split(TEST_DIR, "test"))

    print(f"[INFO] Found PKLs: train/val/test total = {len(pkl_rows)}")

    existing_md5s: Set[str] = set(
        df["md5"].dropna().astype(str).str.strip().tolist()
    )

    missing = [row for row in pkl_rows if row["md5"] not in existing_md5s]

    print(f"[INFO] Missing PKL entries not present in CSV md5 column = {len(missing)}")

    missing_md5s = [row["md5"] for row in missing]

    if missing_md5s:
        print("\nNEW / MISSING MD5 ENTRIES:")
        print("[")
        for md5 in missing_md5s:
            print(f'    "{md5}",')
        print("]")

    new_rows = [build_new_row(df, row) for row in missing]

    if new_rows:
        updated_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    else:
        updated_df = df.copy()

    updated_df.to_csv(OUTPUT_METADATA_PATH, index=False)

    print(f"\n[INFO] Original CSV : {METADATA_PATH}")
    print(f"[INFO] Output CSV   : {OUTPUT_METADATA_PATH}")
    print(f"[INFO] Added rows   : {len(new_rows)}")


if __name__ == "__main__":
    main()