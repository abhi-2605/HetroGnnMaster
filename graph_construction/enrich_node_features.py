"""
graph_construction/enrich_node_features.py

Enrich API nodes with semantic embeddings from LLM.
Replaces one-hot encodings with dense semantic vectors.

Usage:
    python -m graph_construction.enrich_node_features
"""

import pickle
import sys
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import HeteroData

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))


class NodeFeatureEnricher:
    """
    Enrich API node features with LLM semantic embeddings.
    """
    
    def __init__(
        self,
        cache_dir: Path = Path('cache'),
        enriched_dir: Path = Path('cache/enriched')
    ):
        """
        Initialize the enricher.
        
        Args:
            cache_dir: Directory containing original graphs and embeddings
            enriched_dir: Directory to save enriched graphs
        """
        self.cache_dir = Path(cache_dir)
        self.enriched_dir = Path(enriched_dir)
        self.embeddings_path = self.cache_dir / 'api_embeddings.pkl'
        self.api_embeddings = None
        self.embedding_dim = None
        
        # Statistics
        self.stats = {
            'total_graphs_processed': 0,
            'total_api_nodes_enriched': 0,
            'failed_graphs': 0,
            'unknown_apis': 0,
            'train_processed': 0,
            'val_processed': 0,
            'test_processed': 0,
            'original_api_feature_dim': None,
            'enriched_api_feature_dim': None
        }
    
    def setup(self):
        """Setup directories and validate paths."""
        print("="*70)
        print("NODE FEATURE ENRICHMENT - Step 2 of Graph Enrichment")
        print("="*70)
        
        # Check if embeddings exist
        if not self.embeddings_path.exists():
            print(f"❌ Error: API embeddings not found at {self.embeddings_path}")
            print("   Please run llm_setup.py first!")
            sys.exit(1)
        
        # Check if graph directories exist
        train_dir = self.cache_dir / 'train'
        val_dir = self.cache_dir / 'val'
        test_dir = self.cache_dir / 'test'
        
        for split_dir in [train_dir, val_dir, test_dir]:
            if not split_dir.exists():
                print(f"❌ Error: Graph directory not found: {split_dir}")
                sys.exit(1)
        
        # Count graphs
        train_files = list(train_dir.glob('*.pkl'))
        val_files = list(val_dir.glob('*.pkl'))
        test_files = list(test_dir.glob('*.pkl'))
        
        print(f"\n✓ Found graphs to enrich:")
        print(f"  - Train: {len(train_files):,} graphs")
        print(f"  - Val:   {len(val_files):,} graphs")
        print(f"  - Test:  {len(test_files):,} graphs")
        print(f"  - Total: {len(train_files) + len(val_files) + len(test_files):,} graphs")
        
        # Create enriched directories
        for split in ['train', 'val', 'test']:
            enriched_split_dir = self.enriched_dir / split
            enriched_split_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n✓ Enriched graphs will be saved to: {self.enriched_dir}/")
        
        # Check if enriched graphs already exist
        existing_enriched = list(self.enriched_dir.glob('*/*.pkl'))
        if existing_enriched:
            print(f"\n⚠️  Found {len(existing_enriched)} existing enriched graphs")
            response = input("   Overwrite? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(0)
    
    def load_embeddings(self):
        """Load pre-computed API embeddings."""
        print(f"\n{'='*70}")
        print("Loading API Embeddings")
        print(f"{'='*70}")
        
        try:
            with open(self.embeddings_path, 'rb') as f:
                self.api_embeddings = pickle.load(f)
            
            # Get embedding dimension from first embedding
            first_embedding = next(iter(self.api_embeddings.values()))
            self.embedding_dim = len(first_embedding)
            
            print(f"✓ Loaded {len(self.api_embeddings):,} API embeddings")
            print(f"  Embedding dimension: {self.embedding_dim}")
            print(f"  Sample APIs: {list(self.api_embeddings.keys())[:5]}")
            
        except Exception as e:
            print(f"❌ Error loading embeddings: {str(e)}")
            sys.exit(1)
    
    def enrich_api_features(
        self,
        graph: HeteroData,
        graph_id: str
    ) -> HeteroData:
        """
        Enrich API node features with semantic embeddings.
        
        Args:
            graph: Original HeteroData graph
            graph_id: Graph identifier for error messages
        
        Returns:
            Enriched HeteroData graph
        """
        # Check if graph has API nodes
        if 'api' not in graph.node_types:
            return graph
        
        api_store = graph['api']
        
        # Get API names
        api_names = None
        for attr in ['api_names', 'names', 'api_name']:
            if hasattr(api_store, attr):
                api_names = getattr(api_store, attr)
                break
        
        if api_names is None:
            raise ValueError(f"No API names found in graph {graph_id}")
        
        # Get current features
        current_features = api_store.x
        num_apis = len(api_names)
        
        # Record original feature dimension (only once)
        if self.stats['original_api_feature_dim'] is None:
            self.stats['original_api_feature_dim'] = current_features.shape[1]
        
        # Build new feature matrix
        # New features: [embedding(768), timestamp, pid, 5 category flags]
        # Total: 768 + 1 + 1 + 5 = 775 dims
        
        new_features = []
        
        for i, api_name in enumerate(api_names):
            # Get LLM embedding
            if api_name in self.api_embeddings:
                embedding = self.api_embeddings[api_name]
            else:
                # Unknown API - use zero vector
                embedding = np.zeros(self.embedding_dim, dtype=np.float32)
                self.stats['unknown_apis'] += 1
            
            # Get other features from current feature vector
            # Assuming structure: [one_hot/embedding, timestamp, pid, 5_flags]
            # We want to keep: timestamp, pid, 5 flags (last 7 values)
            other_features = current_features[i, -7:].cpu().numpy()
            
            # Concatenate: embedding + other features
            new_feature = np.concatenate([embedding, other_features])
            new_features.append(new_feature)
        
        # Convert to tensor
        new_features = torch.tensor(
            np.array(new_features),
            dtype=torch.float32
        )
        
        # Update graph
        graph['api'].x = new_features
        
        # Record enriched feature dimension (only once)
        if self.stats['enriched_api_feature_dim'] is None:
            self.stats['enriched_api_feature_dim'] = new_features.shape[1]
        
        self.stats['total_api_nodes_enriched'] += num_apis
        
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
        
        # Get input and output directories
        input_dir = self.cache_dir / split_name
        output_dir = self.enriched_dir / split_name
        
        # Get all graph files
        graph_files = sorted(input_dir.glob('*.pkl'))
        
        print(f"Processing {len(graph_files):,} graphs...\n")
        
        processed = 0
        failed = 0
        
        for graph_file in tqdm(graph_files, desc=f"Enriching {split_name}"):
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
                
                # Enrich API features
                enriched_graph = self.enrich_api_features(graph, graph_file.stem)
                
                # Save enriched graph
                output_file = output_dir / graph_file.name
                with open(output_file, 'wb') as f:
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
        """Validate enriched graphs."""
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
            
            # Check API features
            if 'api' in graph.node_types:
                api_features = graph['api'].x
                print(f"\n{split.capitalize()} sample ({graph_files[0].name}):")
                print(f"  API nodes: {api_features.shape[0]}")
                print(f"  Feature dimension: {api_features.shape[1]}")
                print(f"  Feature range: [{api_features.min():.4f}, {api_features.max():.4f}]")
                print(f"  Label: {label}")
                
                # Check if enrichment worked
                if api_features.shape[1] == self.stats['enriched_api_feature_dim']:
                    print(f"  Status: ✓ Correctly enriched")
                else:
                    print(f"  Status: ⚠️  Unexpected dimension")
    
    def save_statistics(self):
        """Save enrichment statistics."""
        stats_path = self.enriched_dir / 'enrichment_stats.pkl'
        
        try:
            with open(stats_path, 'wb') as f:
                pickle.dump(self.stats, f)
            print(f"\n✓ Statistics saved to: {stats_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save statistics: {str(e)}")
    
    def run(self):
        """Run the complete enrichment pipeline."""
        try:
            # Setup
            self.setup()
            
            # Load embeddings
            self.load_embeddings()
            
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
            print(f"\nEnrichment Summary:")
            print(f"  Total graphs processed: {self.stats['total_graphs_processed']:,}")
            print(f"    - Train: {self.stats['train_processed']:,}")
            print(f"    - Val:   {self.stats['val_processed']:,}")
            print(f"    - Test:  {self.stats['test_processed']:,}")
            print(f"  Total API nodes enriched: {self.stats['total_api_nodes_enriched']:,}")
            print(f"  Failed graphs: {self.stats['failed_graphs']}")
            print(f"  Unknown APIs (used zero vector): {self.stats['unknown_apis']}")
            print(f"\nFeature Dimensions:")
            print(f"  Original: {self.stats['original_api_feature_dim']}")
            print(f"  Enriched: {self.stats['enriched_api_feature_dim']} "
                  f"({self.embedding_dim} embedding + 7 other features)")
            print(f"\nOutput: {self.enriched_dir}/")
            
            print(f"\n🔜 Next step: Run edge enrichment")
            print(f"   python -m graph_construction.enrich_edge_features")
            
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
    print("start")
    enricher = NodeFeatureEnricher(
        cache_dir=Path('cache'),
        enriched_dir=Path('cache/enriched')
    )
    enricher.run()


if __name__ == '__main__':
    main()
