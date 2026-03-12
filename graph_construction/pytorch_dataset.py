"""
PyTorch Dataset for Temporal Heterogeneous GNN (TGNN) Ransomware Detection

Preserves HeteroData structure with multiple node/edge types for temporal analysis.

3-CLASS CLASSIFICATION:
    - 0: Ransomware
    - 1: Benign
    - 2: Other Malware (trojan, banker, adware, etc.)
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import HeteroData, Batch
import pickle
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging
import copy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASS_NAMES = {
    0: 'Malware',
    1: 'Benign',
}



class RansomwareGraphDataset(Dataset):
    """
    Dataset for Temporal Heterogeneous Graphs.
    
    Each HeteroData graph contains:
        Node types: ['api', 'file', 'registry', 'process', 'network']
        Edge types: [('api', 'temporal', 'api'), ('api', 'operates_on', 'file'), ...]
        
    Features:
        - API nodes: 775-dim (768 embeddings + 7 features)
        - Edge features: 14-dim temporal features
        - Label: 0=Ransomware, 1=Benign, 2=Other Malware
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        cache_dir: str = 'cache/enriched'
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.cache_dir = Path(cache_dir)
        self.split_dir = self.cache_dir / split
        
        # Load graph file paths
        self.graph_files = sorted(list(self.split_dir.glob('*.pkl')))
        
        if len(self.graph_files) == 0:
            raise ValueError(f"No .pkl files found in {self.split_dir}")
        
        logger.info(f"Loaded {len(self.graph_files)} HeteroData graphs for {split} split")
        
        # Load labels
        self.labels = self._load_labels()
    
    def _load_labels(self) -> Dict[str, int]:
        """Load 3-class labels from metadata CSV."""
        metadata_path = self.data_dir / 'extracted_metadata.csv'
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        df = pd.read_csv(metadata_path)
        
        labels = {}
        for _, row in df.iterrows():
            hash_id = row['md5']
            category_names = str(row['all_category_names']).lower()
            
            if 'benign' in category_names:
                label = 1   # Benign
            else:
                label = 0   # Malware (ransomware + all other malware)

            
            labels[hash_id] = label
        
        logger.info(f"Loaded {len(labels)} labels from metadata CSV")
        
        # Count classes for this split
        split_labels = []
        for graph_file in self.graph_files:
            hash_id = graph_file.stem
            if hash_id in labels:
                split_labels.append(labels[hash_id])
        
        logger.info(f"  {self.split} split:")
        logger.info(f"    - Malware: {sum(1 for l in split_labels if l == 0)}")
        logger.info(f"    - Benign: {sum(1 for l in split_labels if l == 1)}")
        logger.info(f"    - Other malware: {sum(1 for l in split_labels if l == 2)}")
        
        return labels
    
    def __len__(self) -> int:
        return len(self.graph_files)
    
    def __getitem__(self, idx: int) -> Tuple[HeteroData, int]:
        """
        Load HeteroData graph with temporal structure preserved.
        
        Returns:
            Tuple of (HeteroData graph, label)
        """
        graph_path = self.graph_files[idx]
        
        with open(graph_path, 'rb') as f:
            data = pickle.load(f)
        
        # Verify it's HeteroData
        if not isinstance(data, HeteroData):
            raise TypeError(f"Expected HeteroData, got {type(data)}")
        
        # Extract hash ID and get label
        hash_id = graph_path.stem
        
        if hash_id in self.labels:
            label = self.labels[hash_id]
        else:
            logger.warning(f"No label found for {hash_id}, defaulting to 2")
            label = 2
        if self.split == 'train' and label == 1:
            data = self._augment_graph(data)
        # Return graph and label separately
        return data, label
    
    def _augment_graph(self, data, epoch: int = 1) -> HeteroData:
        data = copy.deepcopy(data)

        # 0.05 → 0.15 over 10 epochs, plateau after
        noise_std = min(0.05 + epoch * 0.01, 0.15)   # epoch 1=0.06, epoch 5=0.10, epoch 10+=0.15

        for node_type in data.node_types:
            if hasattr(data[node_type], 'x') and data[node_type].x is not None:
                data[node_type].x = data[node_type].x + \
                    torch.randn_like(data[node_type].x) * noise_std

        # edge drop: 10% → 20% over 10 epochs
        drop_rate = min(0.10 + epoch * 0.01, 0.20)   # epoch 1=0.11, epoch 5=0.15, epoch 10+=0.20
        for edge_type in data.edge_types:
            ei = data[edge_type].edge_index
            if ei is None or ei.size(1) == 0:
                continue
            mask = torch.rand(ei.size(1)) > drop_rate
            data[edge_type].edge_index = ei[:, mask]
            if hasattr(data[edge_type], 'edge_attr') and \
                    data[edge_type].edge_attr is not None:
                data[edge_type].edge_attr = data[edge_type].edge_attr[mask]

        return data


    

    
    def get_statistics(self) -> dict:
        """Get dataset statistics for HeteroData."""
        split_labels = []
        for graph_file in self.graph_files:
            hash_id = graph_file.stem
            if hash_id in self.labels:
                split_labels.append(self.labels[hash_id])
        
        stats = {
            'num_graphs': len(self),
            'num_ransomware': sum(1 for l in split_labels if l == 0),
            'num_benign': sum(1 for l in split_labels if l == 1),
            'num_other_malware': sum(1 for l in split_labels if l == 2),
            'split': self.split,
            'graph_type': 'HeteroData (Temporal)'
        }
        
        # Sample first graph structure
        if len(self) > 0:
            sample, sample_label = self[0]
            stats['node_types'] = sample.node_types
            stats['edge_types'] = sample.edge_types
            
            # Count nodes per type
            for node_type in sample.node_types:
                if hasattr(sample[node_type], 'x'):
                    stats[f'{node_type}_nodes'] = sample[node_type].x.shape[0]
                    stats[f'{node_type}_features'] = sample[node_type].x.shape[1]
        
        return stats


def hetero_collate(batch):
    """Collate function for HeteroData batching with automatic cleanup."""
    graphs = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    all_api_names = []
    for graph in graphs:
        if hasattr(graph['api'], 'api_names'):
            all_api_names.extend(graph['api'].api_names)
    # Collect timestamps from all graphs
    timestamps_dict = {}
    if len(graphs) > 0:
        for node_type in graphs[0].node_types:
            if hasattr(graphs[0][node_type], 'timestamps'):
                all_timestamps = []
                for graph in graphs:
                    if hasattr(graph[node_type], 'timestamps'):
                        all_timestamps.append(graph[node_type].timestamps)
                if all_timestamps:
                    timestamps_dict[node_type] = torch.cat(all_timestamps)
    
    # FIX 1: Pad resource features to 6 dimensions
    for graph in graphs:
        for node_type in ['file', 'registry', 'process', 'network']:
            if node_type in graph.node_types:
                x = graph[node_type].x
                if x.shape[1] < 6:
                    padding = torch.zeros(x.shape[0], 6 - x.shape[1], dtype=x.dtype)
                    graph[node_type].x = torch.cat([x, padding], dim=1)
                elif x.shape[1] > 6:
                    graph[node_type].x = x[:, :6]
    
    # FIX 2: Remove non-tensor attributes but KEEP timestamps
    keep_attrs = ['timestamps', 'timestamp', 'time', 'num_nodes']  # Keep num_nodes too!
    
    for graph in graphs:
        # Clean node stores
        for node_type in graph.node_types:
            store = graph[node_type]
            keys_to_remove = []
            
            for key in list(store.keys()):
                if key in keep_attrs:
                    continue
                try:
                    val = store[key]
                    if not isinstance(val, torch.Tensor):
                        keys_to_remove.append(key)
                except:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                try:
                    del store[key]
                except:
                    pass
        
        # Clean edge stores
        for edge_type in graph.edge_types:
            store = graph[edge_type]
            keys_to_remove = []
            
            for key in list(store.keys()):
                if key in keep_attrs:
                    continue
                try:
                    val = store[key]
                    if not isinstance(val, torch.Tensor):
                        keys_to_remove.append(key)
                except:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                try:
                    del store[key]
                except:
                    pass
    
    # FIX 3: Set num_nodes for each node type (REQUIRED for batching!)
    for graph in graphs:
        for node_type in graph.node_types:
            if hasattr(graph[node_type], 'x'):
                graph[node_type].num_nodes = graph[node_type].x.shape[0]
    
    # Now batch the graphs
    batched_graph = Batch.from_data_list(graphs)
    if all_api_names:
        batched_graph['api'].api_names = all_api_names  # Keep api_names for vocab indexing 
    # ALWAYS attach timestamps_dict (even if empty) to avoid KeyError
    batched_graph.timestamps_dict = timestamps_dict if timestamps_dict else None
    
    return batched_graph, labels








def create_dataloaders(
    data_dir: str = 'data',
    cache_dir: str = 'cache/enriched',
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle_train: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for HeteroData graphs.
    
    Uses custom collate to handle graph-level labels separately.
    
    Args:
        data_dir: Root directory containing extracted_metadata.csv
        cache_dir: Directory with enriched graphs
        batch_size: Number of graphs per batch
        num_workers: Number of parallel workers for data loading
        shuffle_train: Whether to shuffle training data
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    try:
        # Create datasets
        train_dataset = RansomwareGraphDataset(data_dir, 'train', cache_dir)
        val_dataset = RansomwareGraphDataset(data_dir, 'val', cache_dir)
        test_dataset = RansomwareGraphDataset(data_dir, 'test', cache_dir)
        
        # Print statistics
        for dataset in [train_dataset, val_dataset, test_dataset]:
            stats = dataset.get_statistics()
            logger.info(f"\n{stats['split'].upper()} Split (HeteroData):")
            logger.info(f"  Graphs: {stats['num_graphs']}")
            logger.info(f"  Ransomware: {stats['num_ransomware']}")
            logger.info(f"  Benign: {stats['num_benign']}")
            logger.info(f"  Other malware: {stats['num_other_malware']}")
            if 'node_types' in stats:
                logger.info(f"  Node types: {stats['node_types']}")
                logger.info(f"  Edge types: {len(stats['edge_types'])} types")
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            collate_fn=hetero_collate
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=hetero_collate
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=hetero_collate
        )
        
        return train_loader, val_loader, test_loader
    
    except Exception as e:
        logger.error(f"Error creating dataloaders: {str(e)}")
        import traceback
        traceback.print_exc()
        raise  # Re-raise instead of returning None


def validate_dataset(data_dir: str = 'data', cache_dir: str = 'cache/enriched'):
    """Validate HeteroData graphs — tests all recent changes."""
    import copy

    logger.info("\n" + "="*70)
    logger.info("Temporal HeteroData Graph Validation (Full Change Test)")
    logger.info("="*70)

    all_passed = True

    try:
        train_dataset = RansomwareGraphDataset(data_dir, 'train', cache_dir)
        val_dataset   = RansomwareGraphDataset(data_dir, 'val',   cache_dir)
        test_dataset  = RansomwareGraphDataset(data_dir, 'test',  cache_dir)

        # ── TEST 1: 2-Class Labels Only (no class 2 anywhere) ──────────────
        logger.info("\n" + "="*50)
        logger.info("TEST 1: 2-Class Label Verification")
        logger.info("="*50)
        for dataset in [train_dataset, val_dataset, test_dataset]:
            all_labels = [dataset.labels[dataset.graph_files[i].stem]
                          for i in range(len(dataset))
                          if dataset.graph_files[i].stem in dataset.labels]
            unique = set(all_labels)
            has_class2 = 2 in unique
            status = "❌ FAIL — class 2 found!" if has_class2 else "✅ PASS"
            logger.info(f"  [{dataset.split.upper()}] Labels present: {unique} → {status}")
            if has_class2:
                all_passed = False

        # ── TEST 2: Dataset Stats per Split ────────────────────────────────
        logger.info("\n" + "="*50)
        logger.info("TEST 2: Dataset Split Statistics")
        logger.info("="*50)
        for dataset in [train_dataset, val_dataset, test_dataset]:
            stats = dataset.get_statistics()
            logger.info(f"\n  {dataset.split.upper()} split:")
            logger.info(f"    Total graphs  : {stats['num_graphs']}")
            logger.info(f"    Malware (0)   : {stats['num_ransomware']}")
            logger.info(f"    Benign  (1)   : {stats['num_benign']}")
            logger.info(f"    Other   (2)   : {stats['num_other_malware']}  ← should be 0")
            if stats['num_other_malware'] > 0:
                logger.warning("  ❌ FAIL — other_malware count > 0")
                all_passed = False
            else:
                logger.info("  ✅ PASS — no class-2 samples")

        # ── TEST 3: Augmentation ONLY fires on train+benign ────────────────
        logger.info("\n" + "="*50)
        logger.info("TEST 3: Augmentation Guard (train+benign only)")
        logger.info("="*50)

        # Find a benign sample in train
        benign_train_idx = next(
            (i for i in range(len(train_dataset))
             if train_dataset.labels.get(train_dataset.graph_files[i].stem) == 1),
            None
        )
        # Find a malware sample in train
        malware_train_idx = next(
            (i for i in range(len(train_dataset))
             if train_dataset.labels.get(train_dataset.graph_files[i].stem) == 0),
            None
        )
        # Find a benign sample in val
        benign_val_idx = next(
            (i for i in range(len(val_dataset))
             if val_dataset.labels.get(val_dataset.graph_files[i].stem) == 1),
            None
        )

        # 3a — train benign: data SHOULD differ from raw pkl each call
        if benign_train_idx is not None:
            raw_path = train_dataset.graph_files[benign_train_idx]
            with open(raw_path, 'rb') as f:
                raw = pickle.load(f)

            aug1, _ = train_dataset[benign_train_idx]
            aug2, _ = train_dataset[benign_train_idx]

            # Check augmented vs raw differs
            node_type = raw.node_types[0]
            raw_x   = raw[node_type].x
            aug1_x  = aug1[node_type].x
            aug2_x  = aug2[node_type].x

            differs_from_raw   = not torch.allclose(raw_x, aug1_x)
            two_calls_differ   = not torch.allclose(aug1_x, aug2_x)

            logger.info(f"  Train benign augmented vs raw    : {'✅ PASS (different)' if differs_from_raw else '❌ FAIL (identical)'}")
            logger.info(f"  Two train benign calls differ    : {'✅ PASS (stochastic)' if two_calls_differ else '❌ FAIL (same every time)'}")
            if not differs_from_raw or not two_calls_differ:
                all_passed = False
        else:
            logger.warning("  ⚠️  No benign sample found in train split — skipping 3a")

        # 3b — train malware: data should be IDENTICAL to raw pkl
        if malware_train_idx is not None:
            raw_path = train_dataset.graph_files[malware_train_idx]
            with open(raw_path, 'rb') as f:
                raw = pickle.load(f)
            loaded, lbl = train_dataset[malware_train_idx]
            node_type = raw.node_types[0]
            malware_unchanged = torch.allclose(raw[node_type].x, loaded[node_type].x)
            logger.info(f"  Train malware NOT augmented      : {'✅ PASS' if malware_unchanged else '❌ FAIL (was augmented!)'}")
            if not malware_unchanged:
                all_passed = False
        else:
            logger.warning("  ⚠️  No malware sample in train — skipping 3b")

        # 3c — val benign: data should be IDENTICAL to raw pkl (no augmentation)
        if benign_val_idx is not None:
            raw_path = val_dataset.graph_files[benign_val_idx]
            with open(raw_path, 'rb') as f:
                raw = pickle.load(f)
            loaded, lbl = val_dataset[benign_val_idx]
            node_type = raw.node_types[0]
            val_unchanged = torch.allclose(raw[node_type].x, loaded[node_type].x)
            logger.info(f"  Val benign NOT augmented         : {'✅ PASS' if val_unchanged else '❌ FAIL (val was augmented!)'}")
            if not val_unchanged:
                all_passed = False
        else:
            logger.warning("  ⚠️  No benign sample in val — skipping 3c")

        # ── TEST 4: Augmentation Sanity — shape preserved ──────────────────
        logger.info("\n" + "="*50)
        logger.info("TEST 4: Augmentation Shape Preservation")
        logger.info("="*50)
        if benign_train_idx is not None:
            raw_path = train_dataset.graph_files[benign_train_idx]
            with open(raw_path, 'rb') as f:
                raw = pickle.load(f)
            aug, _ = train_dataset[benign_train_idx]

            shape_ok = True
            for nt in raw.node_types:
                if hasattr(raw[nt], 'x') and raw[nt].x is not None:
                    if raw[nt].x.shape[1] != aug[nt].x.shape[1]:
                        logger.error(f"  ❌ Node {nt} feature dim changed: {raw[nt].x.shape} → {aug[nt].x.shape}")
                        shape_ok = False
                    else:
                        logger.info(f"  ✅ Node '{nt}' shape preserved: {aug[nt].x.shape}")
            for et in raw.edge_types:
                raw_edges = raw[et].edge_index.size(1)
                aug_edges = aug[et].edge_index.size(1)
                dropped   = raw_edges - aug_edges
                pct       = dropped / raw_edges * 100 if raw_edges > 0 else 0
                logger.info(f"  Edge '{et[1]}': {raw_edges} → {aug_edges} ({pct:.1f}% dropped)")
                if aug_edges > raw_edges:
                    logger.error(f"  ❌ FAIL — more edges after augmentation!")
                    shape_ok = False
            if not shape_ok:
                all_passed = False

        # ── TEST 5: DataLoader Batching ─────────────────────────────────────
        logger.info("\n" + "="*50)
        logger.info("TEST 5: DataLoader Batching")
        logger.info("="*50)
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=data_dir,
            cache_dir=cache_dir,
            batch_size=4,
            num_workers=0
        )
        batch, labels = next(iter(train_loader))
        logger.info(f"  Batch size        : {batch.num_graphs}")
        logger.info(f"  Labels in batch   : {labels.tolist()}")
        logger.info(f"  Classes in batch  : {[CLASS_NAMES[l] for l in labels.tolist()]}")
        label_set = set(labels.tolist())
        if label_set - {0, 1}:
            logger.error(f"  ❌ FAIL — unexpected labels in batch: {label_set - {0,1}}")
            all_passed = False
        else:
            logger.info(f"  ✅ PASS — only labels 0/1 in batch")

        for node_type in batch.node_types:
            if hasattr(batch[node_type], 'x'):
                logger.info(f"  {node_type} nodes in batch: {batch[node_type].x.shape}")

        # ── FINAL RESULT ────────────────────────────────────────────────────
        logger.info("\n" + "="*70)
        if all_passed:
            logger.info("✅ ALL TESTS PASSED — pytorch_dataset.py changes verified")
        else:
            logger.error("❌ SOME TESTS FAILED — check output above")
        logger.info("="*70)
        return all_passed

    except Exception as e:
        logger.error(f"\n❌ Validation crashed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    validate_dataset()

