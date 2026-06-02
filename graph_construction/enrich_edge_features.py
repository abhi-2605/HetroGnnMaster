"""
graph_construction/enrich_edge_features.py

Enrich temporal edges with advanced temporal features.
Adds 11 new features to existing edge features (handles 2-dim or 3-dim).

Usage:
    python -m graph_construction.enrich_edge_features
"""

import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import HeteroData
from scipy import signal
from scipy.fft import fft

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


class EdgeFeatureEnricher:
    """
    Enrich temporal edge features with advanced temporal patterns.
    """
    
    def __init__(
        self,
        enriched_dir: Path = Path('cache/enriched'),
        window_size: int = 5,
        burst_threshold: float = 0.1
    ):
        """
        Initialize the enricher.
        
        Args:
            enriched_dir: Directory containing node-enriched graphs
            window_size: Window size for computing statistics
            burst_threshold: Time threshold for burst detection (seconds)
        """
        self.enriched_dir = Path(enriched_dir)
        self.window_size = window_size
        self.burst_threshold = burst_threshold
        
        # Statistics
        self.stats = {
            'total_graphs_processed': 0,
            'total_edges_enriched': 0,
            'failed_graphs': 0,
            'train_processed': 0,
            'val_processed': 0,
            'test_processed': 0,
            'original_edge_dims': {},
            'enriched_edge_feature_dim': None,
            'zero_time_deltas': 0,
            'avg_burst_ratio': [],
            'graphs_with_2dim': 0,
            'graphs_with_3dim': 0,
            'graphs_already_enriched': 0
        }
    
    def setup(self):
        """Setup and validate paths."""
        print("="*70)
        print("EDGE FEATURE ENRICHMENT - Step 3 of Graph Enrichment")
        print("="*70)
        
        # Check if enriched directories exist
        train_dir = self.enriched_dir / 'train'
        val_dir = self.enriched_dir / 'val'
        test_dir = self.enriched_dir / 'test'
        
        for split_dir in [train_dir, val_dir, test_dir]:
            if not split_dir.exists():
                print(f"❌ Error: Enriched graph directory not found: {split_dir}")
                print("   Please run enrich_node_features.py first!")
                sys.exit(1)
        
        # Count graphs
        train_files = list(train_dir.glob('*.pkl'))
        val_files = list(val_dir.glob('*.pkl'))
        test_files = list(test_dir.glob('*.pkl'))
        
        print(f"\n✓ Found node-enriched graphs:")
        print(f"  - Train: {len(train_files):,} graphs")
        print(f"  - Val:   {len(val_files):,} graphs")
        print(f"  - Test:  {len(test_files):,} graphs")
        print(f"  - Total: {len(train_files) + len(val_files) + len(test_files):,} graphs")
        
        print(f"\n⚠️  Edge enrichment will OVERWRITE these files")
        print(f"    (Safe to re-run - process is idempotent)")
        response = input("   Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    def compute_speed(self, time_deltas: np.ndarray) -> np.ndarray:
        """
        Compute speed (inverse of time delta).
        
        Args:
            time_deltas: Array of time deltas
        
        Returns:
            Speed values (1/time_delta, capped at max for zero deltas)
        """
        # Avoid division by zero
        speeds = np.where(
            time_deltas > 1e-6,
            1.0 / time_deltas,
            1e6  # Cap at 1 million for near-zero deltas
        )
        return speeds
    
    def compute_acceleration(self, speeds: np.ndarray) -> np.ndarray:
        """
        Compute acceleration (change in speed).
        
        Args:
            speeds: Array of speed values
        
        Returns:
            Acceleration values
        """
        if len(speeds) < 2:
            return np.zeros(len(speeds))
        
        # Compute difference
        acceleration = np.zeros(len(speeds))
        acceleration[1:] = np.diff(speeds)
        return acceleration
    
    def compute_burst_intensity(
        self,
        timestamps: np.ndarray,
        threshold: float
    ) -> np.ndarray:
        """
        Compute burst intensity (number of APIs within threshold window).
        
        Args:
            timestamps: Array of timestamps
            threshold: Time window for burst (seconds)
        
        Returns:
            Burst intensity for each edge
        """
        intensities = np.zeros(len(timestamps))
        
        for i, t in enumerate(timestamps):
            # Count APIs within threshold window before and after
            in_window = np.abs(timestamps - t) <= threshold
            intensities[i] = np.sum(in_window) - 1  # Exclude self
        
        return intensities
    
    def compute_local_burst_ratio(
        self,
        time_deltas: np.ndarray,
        threshold: float,
        window: int
    ) -> np.ndarray:
        """
        Compute local burst ratio (burst APIs / total APIs in window).
        
        Args:
            time_deltas: Array of time deltas
            threshold: Burst threshold
            window: Window size
        
        Returns:
            Local burst ratios
        """
        ratios = np.zeros(len(time_deltas))
        is_burst = time_deltas < threshold
        
        for i in range(len(time_deltas)):
            start = max(0, i - window // 2)
            end = min(len(time_deltas), i + window // 2 + 1)
            
            window_bursts = np.sum(is_burst[start:end])
            window_total = end - start
            
            ratios[i] = window_bursts / window_total if window_total > 0 else 0.0
        
        return ratios
    
    def compute_window_statistics(
        self,
        time_deltas: np.ndarray,
        window: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute sliding window mean and std of time deltas.
        
        Args:
            time_deltas: Array of time deltas
            window: Window size
        
        Returns:
            (window_means, window_stds)
        """
        means = np.zeros(len(time_deltas))
        stds = np.zeros(len(time_deltas))
        
        for i in range(len(time_deltas)):
            start = max(0, i - window + 1)
            end = i + 1
            window_vals = time_deltas[start:end]
            
            means[i] = np.mean(window_vals)
            stds[i] = np.std(window_vals) if len(window_vals) > 1 else 0.0
        
        return means, stds
    
    def compute_periodicity(
        self,
        time_deltas: np.ndarray,
        min_length: int = 10
    ) -> np.ndarray:
        """
        Compute periodicity score using FFT.
        
        Args:
            time_deltas: Array of time deltas
            min_length: Minimum sequence length for FFT
        
        Returns:
            Periodicity scores
        """
        if len(time_deltas) < min_length:
            return np.zeros(len(time_deltas))
        
        # Compute FFT
        try:
            fft_vals = np.abs(fft(time_deltas))
            # Get peak magnitude (exclude DC component)
            peak_magnitude = np.max(fft_vals[1:len(fft_vals)//2])
            # Normalize by mean
            mean_magnitude = np.mean(fft_vals[1:len(fft_vals)//2])
            periodicity_score = peak_magnitude / (mean_magnitude + 1e-6)
            
            # Broadcast to all edges
            return np.full(len(time_deltas), periodicity_score)
        except:
            return np.zeros(len(time_deltas))
    
    def compute_sudden_change(
        self,
        time_deltas: np.ndarray,
        threshold_factor: float = 2.0
    ) -> np.ndarray:
        """
        Detect sudden changes in time deltas.
        
        Args:
            time_deltas: Array of time deltas
            threshold_factor: Factor of std to consider sudden change
        
        Returns:
            Binary indicators of sudden change
        """
        if len(time_deltas) < 2:
            return np.zeros(len(time_deltas))
        
        # Compute differences
        diffs = np.zeros(len(time_deltas))
        diffs[1:] = np.abs(np.diff(time_deltas))
        
        # Threshold based on std
        std = np.std(time_deltas)
        threshold = threshold_factor * std
        
        sudden_changes = (diffs > threshold).astype(np.float32)
        return sudden_changes
    
    def enrich_temporal_edges(
        self,
        graph: HeteroData,
        graph_id: str
    ) -> HeteroData:
        """
        Enrich temporal edges with advanced features.
        
        Args:
            graph: HeteroData graph
            graph_id: Graph identifier
        
        Returns:
            Graph with enriched edges
        """
        # Check if graph has temporal edges
        if ('api', 'temporal', 'api') not in graph.edge_types:
            return graph
        
        edge_store = graph['api', 'temporal', 'api']
        
        # Get current edge features
        if not hasattr(edge_store, 'edge_attr') or edge_store.edge_attr is None:
            # No edge features - skip
            return graph
        
        current_features = edge_store.edge_attr
        num_edges = current_features.shape[0]
        edge_dim = current_features.shape[1]
        
        # Track edge dimensions
        if edge_dim not in self.stats['original_edge_dims']:
            self.stats['original_edge_dims'][edge_dim] = 0
        self.stats['original_edge_dims'][edge_dim] += 1
        
        # Handle different edge dimensions
        if edge_dim == 14:
            # Already enriched - skip (idempotent)
            self.stats['graphs_already_enriched'] += 1
            return graph
        elif edge_dim == 2:
            # Missing is_burst - compute it
            time_deltas = current_features[:, 0].cpu().numpy()
            same_process = current_features[:, 1].cpu().numpy()
            is_burst = (time_deltas < self.burst_threshold).astype(np.float32)
            self.stats['graphs_with_2dim'] += 1
        elif edge_dim == 3:
            # Normal case with all basic features
            time_deltas = current_features[:, 0].cpu().numpy()
            same_process = current_features[:, 1].cpu().numpy()
            is_burst = current_features[:, 2].cpu().numpy()
            self.stats['graphs_with_3dim'] += 1
        # elif edge_dim >= 14:
        #     # Already has extended features - re-enrich (idempotent)
        #     # Extract basic features from first 3 dims
        #     time_deltas = current_features[:, 0].cpu().numpy()
        #     same_process = current_features[:, 1].cpu().numpy()
        #     is_burst = current_features[:, 2].cpu().numpy()
        else:
            # Unexpected dimension - can't process
            print(f"\n⚠️  Unexpected edge dimension {edge_dim} for {graph_id}, skipping")
            return graph
        
        # Count zero deltas
        self.stats['zero_time_deltas'] += np.sum(time_deltas < 1e-6)
        
        # Get edge indices to reconstruct timestamps
        edge_index = edge_store.edge_index
        src_nodes = edge_index[0].cpu().numpy()
        dst_nodes = edge_index[1].cpu().numpy()
        
        # Get API timestamps from node features
        # Assuming timestamp is right after embedding (index 768)
        api_features = graph['api'].x
        embedding_dim = 768  # Standard from all-mpnet-base-v2
        
        if api_features.shape[1] > embedding_dim:
            timestamps = api_features[:, embedding_dim].cpu().numpy()
        else:
            # Fallback - use normalized index as timestamp
            timestamps = np.arange(api_features.shape[0]) / api_features.shape[0]
        
        # Reconstruct edge timestamps (use source node timestamp)
        edge_timestamps = timestamps[src_nodes]
        
        # Get process start and end times
        process_start = np.min(timestamps)
        process_end = np.max(timestamps)
        process_duration = process_end - process_start
        
        # Compute new features
        # 1. Speed
        speeds = self.compute_speed(time_deltas)
        
        # 2. Acceleration
        accelerations = self.compute_acceleration(speeds)
        
        # 3. Burst intensity
        burst_intensities = self.compute_burst_intensity(edge_timestamps, self.burst_threshold)
        
        # 4. Local burst ratio
        local_burst_ratios = self.compute_local_burst_ratio(
            time_deltas, self.burst_threshold, self.window_size
        )
        
        # 5. Time from process start
        time_from_start = edge_timestamps - process_start
        # Normalize
        time_from_start = time_from_start / (process_duration + 1e-6)
        
        # 6. Time to process end
        time_to_end = process_end - edge_timestamps
        # Normalize
        time_to_end = time_to_end / (process_duration + 1e-6)
        
        # 7. Sequence position (normalized)
        seq_positions = np.arange(num_edges) / (num_edges - 1) if num_edges > 1 else np.zeros(num_edges)
        
        # 8 & 9. Window statistics
        window_means, window_stds = self.compute_window_statistics(time_deltas, self.window_size)
        
        # 10. Periodicity
        periodicity_scores = self.compute_periodicity(time_deltas)
        
        # 11. Sudden change
        sudden_changes = self.compute_sudden_change(time_deltas)
        
        # Stack all features (always create 14-dim output)
        new_features = np.column_stack([
            time_deltas,           # 0: original
            same_process,          # 1: original
            is_burst,              # 2: original or computed
            speeds,                # 3: new
            accelerations,         # 4: new
            burst_intensities,     # 5: new
            local_burst_ratios,    # 6: new
            time_from_start,       # 7: new
            time_to_end,           # 8: new
            seq_positions,         # 9: new
            window_means,          # 10: new
            window_stds,           # 11: new
            periodicity_scores,    # 12: new
            sudden_changes         # 13: new
        ])

        # Convert to tensor
        new_features = torch.tensor(new_features, dtype=torch.float32)
        
        # Update graph
        graph['api', 'temporal', 'api'].edge_attr = new_features
        
        # Record enriched dimension (only once)
        if self.stats['enriched_edge_feature_dim'] is None:
            self.stats['enriched_edge_feature_dim'] = new_features.shape[1]
        
        # Track burst ratio for this graph
        avg_burst_ratio = np.mean(local_burst_ratios)
        self.stats['avg_burst_ratio'].append(avg_burst_ratio)
        
        self.stats['total_edges_enriched'] += num_edges
        
        return graph
    
    def process_split(self, split_name: str):
        """
        Process all graphs in a split.
        
        Args:
            split_name: 'train', 'val', or 'test'
        """
        print(f"\n{'='*70}")
        print(f"Processing {split_name.upper()} Split")
        print(f"{'='*70}")
        
        # Get directory
        split_dir = self.enriched_dir / split_name
        
        # Get all graph files
        graph_files = sorted(split_dir.glob('*.pkl'))
        
        print(f"Processing {len(graph_files):,} graphs...\n")
        
        processed = 0
        failed = 0
        
        for graph_file in tqdm(graph_files, desc=f"Enriching {split_name} edges"):
            try:
                # Load graph
                with open(graph_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Extract graph and label
                if isinstance(data, tuple) and len(data) == 2:
                    graph, label = data
                elif isinstance(data, tuple) and len(data) == 1:
                    graph = data[0]
                    label = None
                else:
                    graph = data
                    label = None
                
                # Enrich temporal edges
                enriched_graph = self.enrich_temporal_edges(graph, graph_file.stem)
                
                # Save enriched graph (OVERWRITE)
                with open(graph_file, 'wb') as f:
                    if label is not None:
                        pickle.dump((enriched_graph, label), f, protocol=pickle.HIGHEST_PROTOCOL)
                    else:
                        pickle.dump(enriched_graph, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                processed += 1
                self.stats['total_graphs_processed'] += 1
                
            except Exception as e:
                print(f"\n⚠️  Failed to process {graph_file.name}: {str(e)}")
                failed += 1
                self.stats['failed_graphs'] += 1
                continue
        
        # Update split-specific stats
        self.stats[f'{split_name}_processed'] = processed
        
        print(f"\n✓ {split_name.capitalize()} split complete:")
        print(f"  Processed: {processed:,}")
        print(f"  Failed: {failed}")
    
    def validate_enrichment(self):
        """Validate enriched edges."""
        print(f"\n{'='*70}")
        print("Validation")
        print(f"{'='*70}")
        
        # Load one enriched graph from each split and validate
        for split in ['train', 'val', 'test']:
            split_dir = self.enriched_dir / split
            graph_files = list(split_dir.glob('*.pkl'))
            
            if not graph_files:
                print(f"⚠️  No graphs found in {split} split")
                continue
            
            # Load first graph
            with open(graph_files[0], 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, tuple):
                graph, label = data
            else:
                graph = data
                label = None
            
            # Check edge features
            if ('api', 'temporal', 'api') in graph.edge_types:
                edge_features = graph['api', 'temporal', 'api'].edge_attr
                print(f"\n{split.capitalize()} sample ({graph_files[0].name}):")
                print(f"  Temporal edges: {edge_features.shape[0]}")
                print(f"  Edge feature dimension: {edge_features.shape[1]}")
                print(f"  Feature range: [{edge_features.min():.4f}, {edge_features.max():.4f}]")
                
                # Show feature breakdown
                if edge_features.shape[1] >= 14:
                    print(f"  Feature statistics:")
                    print(f"    Time delta (avg): {edge_features[:, 0].mean():.6f}")
                    print(f"    Speed (avg): {edge_features[:, 3].mean():.2f}")
                    print(f"    Burst intensity (avg): {edge_features[:, 5].mean():.2f}")
                    print(f"    Local burst ratio (avg): {edge_features[:, 6].mean():.4f}")
                    print(f"    Periodicity (avg): {edge_features[:, 12].mean():.4f}")
                
                # Check if enrichment worked
                if edge_features.shape[1] == 14:
                    print(f"  Status: ✓ Correctly enriched (14 dims)")
                else:
                    print(f"  Status: ⚠️  Unexpected dimension ({edge_features.shape[1]})")
    
    def save_statistics(self):
        """Save enrichment statistics."""
        stats_path = self.enriched_dir / 'edge_enrichment_stats.pkl'
        
        # Compute summary statistics
        if self.stats['avg_burst_ratio']:
            self.stats['mean_burst_ratio'] = np.mean(self.stats['avg_burst_ratio'])
            self.stats['std_burst_ratio'] = np.std(self.stats['avg_burst_ratio'])
        
        try:
            with open(stats_path, 'wb') as f:
                pickle.dump(self.stats, f)
            print(f"\n✓ Statistics saved to: {stats_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save statistics: {str(e)}")
    
    def run(self):
        """Run the complete edge enrichment pipeline."""
        try:
            # Setup
            self.setup()
            
            # Process each split
            for split in ['train', 'val', 'test']:
                self.process_split(split)
            
            # Validate
            self.validate_enrichment()
            
            # Save statistics
            self.save_statistics()
            
            # Final summary
            print(f"\n{'='*70}")
            print("✅ COMPLETE!")
            print(f"{'='*70}")
            print(f"\nEdge Enrichment Summary:")
            print(f"  Total graphs processed: {self.stats['total_graphs_processed']:,}")
            print(f"    - Train: {self.stats['train_processed']:,}")
            print(f"    - Val:   {self.stats['val_processed']:,}")
            print(f"    - Test:  {self.stats['test_processed']:,}")
            print(f"  Total edges enriched: {self.stats['total_edges_enriched']:,}")
            print(f"  Failed graphs: {self.stats['failed_graphs']}")
            
            print(f"\nOriginal Edge Dimensions Found:")
            for dim, count in sorted(self.stats['original_edge_dims'].items()):
                print(f"  {dim}-dim edges: {count} graphs")
            
            print(f"\nProcessing Details:")
            print(f"  Graphs with 2-dim edges (is_burst computed): {self.stats['graphs_with_2dim']}")
            print(f"  Graphs with 3-dim edges (normal): {self.stats['graphs_with_3dim']}")
            print(f"  Graphs already enriched (skipped): {self.stats['graphs_already_enriched']}")
            
            print(f"\nEdge Feature Dimension:")
            print(f"  Enriched: {self.stats['enriched_edge_feature_dim']}")
            print(f"    (3 basic + 11 new temporal features)")
            
            print(f"\nTemporal Statistics:")
            print(f"  Zero time deltas: {self.stats['zero_time_deltas']:,}")
            if 'mean_burst_ratio' in self.stats:
                print(f"  Mean burst ratio: {self.stats['mean_burst_ratio']:.4f}")
                print(f"  Std burst ratio: {self.stats['std_burst_ratio']:.4f}")
            
            print(f"\n✅ Graphs in cache/enriched/ now have:")
            print(f"   - Enriched API nodes: 775-dim (768 embedding + 7 features)")
            print(f"   - Enriched temporal edges: 14-dim (3 basic + 11 temporal)")
            print(f"\n🔜 Next step: Create PyTorch Dataset and start training!")
            print(f"   python -m graph_construction.pytorch_dataset")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ Unexpected error: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    """Main entry point."""
    enricher = EdgeFeatureEnricher(
        enriched_dir=Path('cache/enriched'),
        window_size=5,
        burst_threshold=0.1
    )
    enricher.run()


if __name__ == '__main__':
    main()
