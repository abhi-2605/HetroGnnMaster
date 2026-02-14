"""
graph_construction/llm_setup.py

Generate and cache semantic embeddings for all unique API names using
sentence-transformers. This is Step 1 of graph enrichment.

Usage:
    python -m graph_construction.llm_setup
"""

import pickle
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set
import numpy as np
import torch
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import config
try:
    import config
    LLM_MODEL_NAME = getattr(config, 'LLM_MODEL_NAME', 'all-mpnet-base-v2')
except ImportError:
    print("⚠️  Warning: config.py not found, using default model")
    LLM_MODEL_NAME = 'all-mpnet-base-v2'

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ Error: sentence-transformers not installed")
    print("Install with: pip install sentence-transformers")
    sys.exit(1)


class LLMEmbeddingGenerator:
    """
    Generate semantic embeddings for API names using sentence-transformers.
    """
    
    def __init__(
        self,
        model_name: str = None,
        cache_dir: Path = Path('cache')
    ):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: HuggingFace model name (default from config)
            cache_dir: Directory containing train/val/test subdirectories
        """
        self.model_name = model_name or LLM_MODEL_NAME
        self.cache_dir = Path(cache_dir)
        self.embeddings_path = self.cache_dir / 'api_embeddings.pkl'
        self.model = None
        self.embedding_dim = None
        
        # Statistics
        self.stats = {
            'total_graphs_scanned': 0,
            'total_apis_seen': 0,
            'unique_apis': 0,
            'api_frequency': Counter(),
            'failed_graphs': 0,
            'embedding_dim': None,
            'train_graphs': 0,
            'val_graphs': 0,
            'test_graphs': 0
        }
    
    def setup(self):
        """Setup directories and validate paths."""
        print("="*70)
        print("LLM EMBEDDING SETUP - Step 1 of Graph Enrichment")
        print("="*70)
        
        # Check if cache directory exists
        if not self.cache_dir.exists():
            print(f"❌ Error: Cache directory not found: {self.cache_dir}")
            print("   Please run data processing first!")
            sys.exit(1)
        
        # Check for train/val/test subdirectories
        train_dir = self.cache_dir / 'train'
        val_dir = self.cache_dir / 'val'
        test_dir = self.cache_dir / 'test'
        
        if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
            print(f"❌ Error: Expected subdirectories not found in {self.cache_dir}")
            print("   Expected structure:")
            print("   cache/")
            print("   ├── train/")
            print("   ├── val/")
            print("   └── test/")
            sys.exit(1)
        
        # Count graph files in each directory
        train_files = list(train_dir.glob('*.pkl'))
        val_files = list(val_dir.glob('*.pkl'))
        test_files = list(test_dir.glob('*.pkl'))
        
        self.stats['train_graphs'] = len(train_files)
        self.stats['val_graphs'] = len(val_files)
        self.stats['test_graphs'] = len(test_files)
        
        total_files = len(train_files) + len(val_files) + len(test_files)
        
        if total_files == 0:
            print(f"❌ Error: No .pkl graph files found in cache subdirectories")
            sys.exit(1)
        
        print(f"\n✓ Found graph files:")
        print(f"  - Train: {len(train_files):,} graphs")
        print(f"  - Val:   {len(val_files):,} graphs")
        print(f"  - Test:  {len(test_files):,} graphs")
        print(f"  - Total: {total_files:,} graphs")
        
        # Check if embeddings already exist
        if self.embeddings_path.exists():
            response = input(f"\n⚠️  Embeddings already exist at {self.embeddings_path}\n"
                           f"   Overwrite? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                sys.exit(0)
    
    def load_model(self):
        """Load sentence-transformer model."""
        print(f"\n{'='*70}")
        print(f"Loading Model: {self.model_name}")
        print(f"{'='*70}")
        
        try:
            print("Downloading/loading model (first time may take a few minutes)...")
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.stats['embedding_dim'] = self.embedding_dim
            
            print(f"✓ Model loaded successfully")
            print(f"  - Model: {self.model_name}")
            print(f"  - Embedding dimension: {self.embedding_dim}")
            print(f"  - Device: {self.model.device}")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            sys.exit(1)
    
    def collect_api_names(self) -> Set[str]:
        """
        Scan all graph files and collect unique API names.
        
        Returns:
            Set of unique API name strings
        """
        print(f"\n{'='*70}")
        print("Collecting Unique API Names from All Graphs")
        print(f"{'='*70}")
        
        unique_apis = set()
        
        # Get all graph files from all splits
        train_files = list((self.cache_dir / 'train').glob('*.pkl'))
        val_files = list((self.cache_dir / 'val').glob('*.pkl'))
        test_files = list((self.cache_dir / 'test').glob('*.pkl'))
        
        all_graph_files = train_files + val_files + test_files
        
        print(f"Scanning {len(all_graph_files):,} graph files...")
        print(f"  (Train: {len(train_files):,}, Val: {len(val_files):,}, Test: {len(test_files):,})\n")
        
        for pkl_file in tqdm(all_graph_files, desc="Scanning graphs"):
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Extract graph and label
                if isinstance(data, tuple) and len(data) == 2:
                    graph, label = data
                elif isinstance(data, tuple) and len(data) == 1:
                    graph = data[0]
                else:
                    graph = data
                
                self.stats['total_graphs_scanned'] += 1
                
                # Get API names from graph
                if hasattr(graph, 'node_types') and 'api' in graph.node_types:
                    api_store = graph['api']
                    
                    # Check various possible attribute names
                    api_names = None
                    for attr in ['api_names', 'names', 'api_name']:
                        if hasattr(api_store, attr):
                            api_names = getattr(api_store, attr)
                            break
                    
                    if api_names is not None:
                        # Handle both list and tensor
                        if isinstance(api_names, list):
                            api_list = api_names
                        elif hasattr(api_names, 'tolist'):
                            api_list = api_names.tolist()
                        else:
                            continue
                        
                        # Add to unique set and count frequency
                        for api in api_list:
                            if api and isinstance(api, str) and api.strip():
                                unique_apis.add(api)
                                self.stats['api_frequency'][api] += 1
                                self.stats['total_apis_seen'] += 1
                
            except Exception as e:
                print(f"\n⚠️  Warning: Failed to process {pkl_file.name}: {str(e)}")
                self.stats['failed_graphs'] += 1
                continue
        
        self.stats['unique_apis'] = len(unique_apis)
        
        print(f"\n{'='*70}")
        print("Collection Statistics:")
        print(f"{'='*70}")
        print(f"  Graphs scanned:       {self.stats['total_graphs_scanned']:,}")
        print(f"  Total API calls:      {self.stats['total_apis_seen']:,}")
        print(f"  Unique API names:     {self.stats['unique_apis']:,}")
        print(f"  Failed graphs:        {self.stats['failed_graphs']}")
        
        if self.stats['unique_apis'] == 0:
            print("\n❌ Error: No API names found in graphs!")
            print("   Check if graphs have 'api_names' attribute")
            sys.exit(1)
        
        # Show top 15 most frequent APIs
        print(f"\nTop 15 Most Frequent APIs:")
        for idx, (api, count) in enumerate(self.stats['api_frequency'].most_common(15), 1):
            print(f"  {idx:2}. {api:45} : {count:,} times")
        
        return unique_apis
    
    def generate_embeddings(self, api_names: Set[str]) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for all API names.
        
        Args:
            api_names: Set of unique API name strings
        
        Returns:
            Dictionary mapping API name → embedding vector
        """
        print(f"\n{'='*70}")
        print("Generating Embeddings")
        print(f"{'='*70}")
        
        api_list = sorted(list(api_names))
        embeddings_dict = {}
        
        # Batch processing for efficiency
        batch_size = 32
        total_batches = (len(api_list) + batch_size - 1) // batch_size
        
        print(f"Processing {len(api_list)} APIs in batches of {batch_size}...")
        print(f"Estimated time: ~{len(api_list) / 30:.1f} seconds\n")
        
        try:
            with tqdm(total=len(api_list), desc="Generating embeddings") as pbar:
                for i in range(0, len(api_list), batch_size):
                    batch = api_list[i:i + batch_size]
                    
                    # Generate embeddings for batch
                    embeddings = self.model.encode(
                        batch,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        batch_size=batch_size
                    )
                    
                    # Store in dictionary
                    for api_name, embedding in zip(batch, embeddings):
                        embeddings_dict[api_name] = embedding
                    
                    pbar.update(len(batch))
        
        except Exception as e:
            print(f"\n❌ Error during embedding generation: {str(e)}")
            sys.exit(1)
        
        print(f"\n✓ Successfully generated {len(embeddings_dict)} embeddings")
        
        # Validate embeddings
        sample_api = api_list[0]
        sample_emb = embeddings_dict[sample_api]
        print(f"\nValidation:")
        print(f"  Sample API: '{sample_api}'")
        print(f"  Embedding shape: {sample_emb.shape}")
        print(f"  Embedding dtype: {sample_emb.dtype}")
        print(f"  Embedding range: [{sample_emb.min():.4f}, {sample_emb.max():.4f}]")
        print(f"  Embedding mean: {sample_emb.mean():.4f}")
        print(f"  Embedding std: {sample_emb.std():.4f}")
        
        return embeddings_dict
    
    def save_embeddings(self, embeddings: Dict[str, np.ndarray]):
        """
        Save embeddings to disk.
        
        Args:
            embeddings: Dictionary of API name → embedding
        """
        print(f"\n{'='*70}")
        print("Saving Embeddings")
        print(f"{'='*70}")
        
        try:
            # Save embeddings
            with open(self.embeddings_path, 'wb') as f:
                pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            file_size_kb = self.embeddings_path.stat().st_size / 1024
            file_size_mb = file_size_kb / 1024
            
            print(f"✓ Embeddings saved to: {self.embeddings_path}")
            print(f"  File size: {file_size_mb:.2f} MB ({file_size_kb:.1f} KB)")
            print(f"  Total embeddings: {len(embeddings):,}")
            print(f"  Embedding dimension: {self.embedding_dim}")
            
            # Test load to verify
            with open(self.embeddings_path, 'rb') as f:
                test_load = pickle.load(f)
            print(f"  Verification: ✓ File loads correctly")
            
        except Exception as e:
            print(f"❌ Error saving embeddings: {str(e)}")
            sys.exit(1)
    
    def save_statistics(self):
        """Save collection statistics."""
        stats_path = self.cache_dir / 'embedding_stats.pkl'
        
        try:
            with open(stats_path, 'wb') as f:
                pickle.dump(self.stats, f)
            print(f"✓ Statistics saved to: {stats_path}")
        except Exception as e:
            print(f"⚠️  Warning: Could not save statistics: {str(e)}")
    
    def run(self):
        """Run the complete embedding generation pipeline."""
        try:
            # Setup
            self.setup()
            
            # Load model
            self.load_model()
            
            # Collect API names
            api_names = self.collect_api_names()
            
            # Generate embeddings
            embeddings = self.generate_embeddings(api_names)
            
            # Save
            self.save_embeddings(embeddings)
            self.save_statistics()
            
            # Final summary
            print(f"\n{'='*70}")
            print("✅ COMPLETE!")
            print(f"{'='*70}")
            print(f"\nSummary:")
            print(f"  Processed splits: train ({self.stats['train_graphs']:,}), "
                  f"val ({self.stats['val_graphs']:,}), test ({self.stats['test_graphs']:,})")
            print(f"  Total graphs: {self.stats['total_graphs_scanned']:,}")
            print(f"  Unique APIs: {self.stats['unique_apis']:,}")
            print(f"  Embedding dimension: {self.embedding_dim}")
            print(f"  Output: {self.embeddings_path}")
            
            print(f"\n🔜 Next step: Run node enrichment")
            print(f"   python -m graph_construction.enrich_node_features")
            
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
    print(f"Using model from config: {LLM_MODEL_NAME}")
    
    generator = LLMEmbeddingGenerator(
        model_name=LLM_MODEL_NAME,
        cache_dir=Path('cache')
    )
    generator.run()


if __name__ == '__main__':
    main()
