"""
Data Loader for TGNN Ransomware Project

- Reads train/val/test split files
- Maps sample IDs → JSON paths
- For each sample:
    * parse_cape_json.load_and_parse_json_file()
    * extract_resources.extract_resources()
    * graph_builder.build_graph()
- Skips samples where:
    * JSON file does not exist
    * JSON file is empty / malformed
    * API sequence parsing returns None or too short
- Optionally caches built graphs to disk for faster reuse
"""

from pathlib import Path
from typing import List, Dict, Optional, Any
import pickle

from torch.utils.data import Dataset

from config import (
    RAW_SAMPLE_DIR,
    SPLITS_DIR,
    GRAPH_CACHE_DIR,
)

from .parse_cape_json import load_and_parse_json_file
from .extract_resources import extract_resources
from .graph_builder import build_graph


class MalwareGraphDataset(Dataset):
    """
    PyTorch Dataset that loads graphs for a given split.

    Split files:
      data/splits/train_samples.txt
      data/splits/val_samples.txt
      data/splits/test_samples.txt

    Each line = sample_id (MD5 or filename without .json)

    Behavior:
      - Iterates over all sample_ids in the split
      - For each sample:
          * If cached graph exists → load from cache
          * Else:
              - Load + parse JSON
              - If parsing fails → SKIP this sample
              - Extract resources
              - Build graph
              - Save to cache
      - __len__ = number of successfully usable samples
    """

    def __init__(
        self,
        split: str = "train",
        use_cache: bool = True,
        rebuild_cache: bool = False,
        max_samples: Optional[int] = None,
    ):
        assert split in {"train", "val", "test"}, f"Invalid split: {split}"
        self.split = split
        self.use_cache = use_cache
        self.rebuild_cache = rebuild_cache
        self.max_samples = max_samples

        self.raw_dir = Path(RAW_SAMPLE_DIR)
        self.SPLITS_DIR = Path(SPLITS_DIR)
        self.cache_dir = Path(GRAPH_CACHE_DIR) / split
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load sample IDs from split file
        split_file = self.SPLITS_DIR / f"{split}_samples.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with split_file.open("r") as f:
            sample_ids = [line.strip() for line in f if line.strip()]

        if self.max_samples is not None:
            sample_ids = sample_ids[: self.max_samples]

        self.sample_ids: List[str] = sample_ids

        # We will lazily build graphs and keep track of which indices are valid
        self._graph_index_map: List[int] = []  # maps dataset idx → sample_ids index

        print(f"✓ Initialized MalwareGraphDataset(split='{split}')")
        print(f"  Total samples listed in split: {len(self.sample_ids)}")

        # Optionally pre-scan and build cache
        self._initialize_graph_index()

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _json_path_for_sample(self, sample_id: str) -> Path:
        """
        Map sample_id to JSON file path.

        Assumes RAW_SAMPLE_DIR has structure:
          RAW_SAMPLE_DIR/
            <category>/
              <sample_id>.json
        or all files directly inside RAW_SAMPLE_DIR.
        """
        # Try direct path first: RAW_SAMPLE_DIR/**/<sample_id>.json
        matches = list(self.raw_dir.rglob(f"{sample_id}.json"))
        if matches:
            return matches[0]
        # Fallback: assume it's directly under RAW_SAMPLE_DIR
        return self.raw_dir / f"{sample_id}.json"

    def _cache_path_for_sample(self, sample_id: str) -> Path:
        return self.cache_dir / f"{sample_id}.pkl"

    def _build_graph_for_sample(self, sample_id: str) -> Optional[Any]:
        """
        Full pipeline for ONE sample:
          JSON → API sequence → resources → graph

        Returns:
          - HeteroData graph if successful
          - None if file missing / empty / parse failed
        """
        json_path = self._json_path_for_sample(sample_id)

        # 1) Check JSON existence
        if not json_path.exists():
            # File missing → skip
            print(f"⚠️  JSON not found for sample {sample_id}, skipping.")
            return None

        # 2) Parse API sequence (parse_cape_json already handles empty / malformed)
        api_sequence = load_and_parse_json_file(json_path)

        if not api_sequence:
            # None or empty list → skip
            print(f"⚠️  Empty/invalid API sequence for {sample_id}, skipping.")
            return None

        # 3) Extract resources
        resources = extract_resources(api_sequence)

        # 4) Build graph
        try:
            graph = build_graph(api_sequence, resources)
        except Exception as e:
            print(f"⚠️  Error building graph for {sample_id}: {e}")
            return None

        return graph
    
    

    def _initialize_graph_index(self):
        """
        Scan through all sample_ids, build or verify cache, and
        record which samples are usable (graph successfully built).

        This is where we ensure:
          - Empty files / parse failures are skipped
          - DataLoader still iterates over all valid samples
        """
        usable_indices: List[int] = []

        try:
            from tqdm import tqdm
            HAS_TQDM = True
        except ImportError:
            HAS_TQDM = False
            print("⚠️  Install tqdm for progress bar: pip install tqdm")

        print(f"🔍 Scanning split='{self.split}' samples to build/verify graphs...")

        iterator = tqdm(enumerate(self.sample_ids), total=len(self.sample_ids), desc="Processing") if HAS_TQDM else enumerate(self.sample_ids)

        for idx, sample_id in enumerate(self.sample_ids):
            cache_path = self._cache_path_for_sample(sample_id)

            if self.use_cache and cache_path.exists() and not self.rebuild_cache:
                # Cached graph exists → mark as usable
                usable_indices.append(idx)
                continue

            # Need to (re)build graph
            graph = self._build_graph_for_sample(sample_id)

            if graph is None:
                # Skip this sample
                continue

            # Save to cache if enabled
            if self.use_cache:
                try:
                    with cache_path.open("wb") as f:
                        pickle.dump(graph, f)
                except Exception as e:
                    print(f"⚠️  Failed to cache graph for {sample_id}: {e}")
                    # Still consider it usable in-memory
                    usable_indices.append(idx)
                    continue

            usable_indices.append(idx)

        self._graph_index_map = usable_indices

        print(f"✓ Usable graphs in split '{self.split}': {len(self._graph_index_map)}")
        skipped = len(self.sample_ids) - len(self._graph_index_map)
        if skipped > 0:
            print(f"  (Skipped {skipped} samples: missing/empty/invalid JSON or graph build error)")

    # --------------------------------------------------------------------- #
    # PyTorch Dataset interface
    # --------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self._graph_index_map)

    def __getitem__(self, idx: int) -> Any:
        """
        Return graph and label for a given dataset index.

        Dataset index → sample_ids index → sample_id → cached graph / build on the fly
        """
        real_idx = self._graph_index_map[idx]
        sample_id = self.sample_ids[real_idx]
        cache_path = self._cache_path_for_sample(sample_id)

        # Load from cache if possible
        if self.use_cache and cache_path.exists():
            with cache_path.open("rb") as f:
                graph = pickle.load(f)
        else:
            graph = self._build_graph_for_sample(sample_id)
            if graph is None:
                # This should be rare because we filtered in _initialize_graph_index,
                # but handle gracefully: raise or return None.
                raise RuntimeError(f"Unexpected: sample {sample_id} became unusable at __getitem__ time.")

        # Attach label if you have it in metadata later (for now, just return graph)
        return graph


# ------------------------------------------------------------------------- #
# Simple CLI test
# ------------------------------------------------------------------------- #

if __name__ == "__main__":
    """
    Quick test:

        python -m data_processing.data_loader train
        python -m data_processing.data_loader val
        python -m data_processing.data_loader test
    """
    import sys

    split = sys.argv[1] if len(sys.argv) > 1 else "train"
    dataset = MalwareGraphDataset(split=split, use_cache=True, rebuild_cache=False, max_samples=None)

    print("\nSample access test:")
    if len(dataset) > 0:
        g0 = dataset[0]
        print(f"First graph node types: {g0.node_types}")
        print(f"First graph has {sum(g0[node].num_nodes for node in g0.node_types)} nodes.")
    else:
        print("No usable graphs in this split.")
