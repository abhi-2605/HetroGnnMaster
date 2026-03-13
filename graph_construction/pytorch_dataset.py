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
import random
import copy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASS_NAMES = {
    0: 'Ransomware',
    1: 'Benign',
    2: 'Other Malware'
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
        self.enable_augmentation = (split == 'train')
        
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
            
            if 'ransomware' in category_names:
                label = 0  # Ransomware
            elif 'benign' in category_names:
                label = 1  # Benign
            else:
                label = 2  # Other malware
            
            labels[hash_id] = label
        
        logger.info(f"Loaded {len(labels)} labels from metadata CSV")
        
        # Count classes for this split
        split_labels = []
        for graph_file in self.graph_files:
            hash_id = graph_file.stem
            if hash_id in labels:
                split_labels.append(labels[hash_id])
        
        logger.info(f"  {self.split} split:")
        logger.info(f"    - Ransomware: {sum(1 for l in split_labels if l == 0)}")
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

        if not isinstance(data, HeteroData):
            raise TypeError(f"Expected HeteroData, got {type(data)}")

        hash_id = graph_path.stem

        if hash_id in self.labels:
            label = self.labels[hash_id]
        else:
            logger.warning(f"No label found for {hash_id}, defaulting to 2")
            label = 2

        if self.enable_augmentation and label == 1:
            if random.random() < 0.7:
                data = self.augment_benign_graph(data)

        return data, label

    def augment_benign_graph(self, data: HeteroData) -> HeteroData:
        """
        Light augmentation for benign training graphs.
        Designed to reduce memorization without destroying semantics.
        """
        data = copy.deepcopy(data)

        resource_node_types = ['file', 'registry', 'process', 'network']

        # 1) Small Gaussian noise + light feature masking on resource nodes
        for nt in resource_node_types:
            if nt in data.node_types and hasattr(data[nt], 'x'):
                x = data[nt].x
                if isinstance(x, torch.Tensor) and x.numel() > 0:
                    noise = 0.01 * torch.randn_like(x)
                    x = x + noise

                    mask = (torch.rand_like(x) > 0.05).float()
                    x = x * mask

                    data[nt].x = x

        # 2) Light dropout on non-temporal edges only
        edge_types_to_drop = [
            ('api', 'operates_on', 'file'),
            ('api', 'modifies', 'registry'),
            ('process', 'executes', 'api'),
            ('api', 'connects_to', 'network'),
        ]

        for et in edge_types_to_drop:
            if et in data.edge_types:
                store = data[et]
                if hasattr(store, 'edge_index') and isinstance(store.edge_index, torch.Tensor):
                    E = store.edge_index.size(1)
                    if E > 0:
                        keep_mask = torch.rand(E, device=store.edge_index.device) > 0.08
                        if keep_mask.sum() == 0:
                            keep_mask[torch.randint(0, E, (1,), device=keep_mask.device)] = True

                        store.edge_index = store.edge_index[:, keep_mask]

                        if hasattr(store, 'edge_attr') and isinstance(store.edge_attr, torch.Tensor):
                            if store.edge_attr.size(0) == E:
                                store.edge_attr = store.edge_attr[keep_mask]

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
        
        # Sample first graph structure without triggering augmentation
        if len(self.graph_files) > 0:
            with open(self.graph_files[0], 'rb') as f:
                sample = pickle.load(f)

            if isinstance(sample, HeteroData):
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
                    padding = torch.zeros(x.shape[0], 6 - x.shape[1], dtype=x.dtype, device=x.device)
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
    """Validate HeteroData graphs."""
    logger.info("\n" + "="*70)
    logger.info("Temporal HeteroData Graph Validation")
    logger.info("="*70)
    
    try:
        train_dataset = RansomwareGraphDataset(data_dir, 'train', cache_dir)
        val_dataset = RansomwareGraphDataset(data_dir, 'val', cache_dir)
        test_dataset = RansomwareGraphDataset(data_dir, 'test', cache_dir)
        
        for dataset in [train_dataset, val_dataset, test_dataset]:
            logger.info(f"\n{'='*50}")
            logger.info(f"Validating {dataset.split.upper()} split")
            logger.info('='*50)
            
            stats = dataset.get_statistics()
            logger.info(f"Graphs: {stats['num_graphs']}")
            logger.info(f"Ransomware: {stats['num_ransomware']}")
            logger.info(f"Benign: {stats['num_benign']}")
            logger.info(f"Other malware: {stats['num_other_malware']}")
            
            # Inspect first 2 samples
            for i in range(min(2, len(dataset))):
                data, label = dataset[i]
                
                logger.info(f"\n  Sample {i+1}: {CLASS_NAMES[label]}")
                logger.info(f"    Node types: {data.node_types}")
                logger.info(f"    Edge types: {len(data.edge_types)} types")
                
                # Show node counts
                for node_type in data.node_types:
                    if hasattr(data[node_type], 'x'):
                        logger.info(f"      {node_type}: {data[node_type].x.shape[0]} nodes, {data[node_type].x.shape[1]}-dim")
                
                # Show temporal edges
                for edge_type in data.edge_types:
                    if 'temporal' in edge_type[1]:
                        num_edges = data[edge_type].edge_index.shape[1]
                        logger.info(f"      Temporal edges: {num_edges}")
                        if hasattr(data[edge_type], 'edge_attr'):
                            logger.info(f"        Edge features: {data[edge_type].edge_attr.shape[1]}-dim")
                        break
        
        # Test DataLoader batching
        logger.info("\n" + "="*70)
        logger.info("Testing HeteroData Batching")
        logger.info("="*70)
        
        train_loader, val_loader, test_loader = create_dataloaders(
            data_dir=data_dir,
            cache_dir=cache_dir,
            batch_size=4,
            num_workers=0
        )
        
        # batch = next(iter(train_loader))
        batch, labels = next(iter(train_loader))
        logger.info(f"\nBatch info:")
        logger.info(f"  Batch size: {batch.num_graphs}")
        logger.info(f"  Type: {type(batch)}")
        logger.info(f"  Node types: {batch.node_types}")

        logger.info(f"  Labels: {labels.tolist()}")
        logger.info(f"  Class distribution: {[CLASS_NAMES[l] for l in labels.tolist()]}")
        
        # Show batched node counts
        for node_type in batch.node_types:
            if hasattr(batch[node_type], 'x'):
                logger.info(f"  {node_type} nodes in batch: {batch[node_type].x.shape[0]}")
        
        logger.info("\n✅ HeteroData validation complete!")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Validation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    validate_dataset()