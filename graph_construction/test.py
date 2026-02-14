# # """
# # Debug script to inspect HeteroData graph structure and identify integer keys.
# # Run this BEFORE validation to understand what's stored.
# # """

# # import pickle
# # from pathlib import Path
# # import torch
# # from torch_geometric.data import HeteroData
# # import logging

# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)


# # def debug_graph_structure(data_dir='data', cache_dir='cache/enriched'):
# #     """
# #     Inspect HeteroData graphs to identify all stored attributes.
# #     """
# #     logger.info("="*70)
# #     logger.info("DEBUG: Inspecting HeteroData Graph Structure")
# #     logger.info("="*70)
    
# #     cache_path = Path(cache_dir) / 'train'
# #     graph_files = sorted(list(cache_path.glob('*.pkl')))[:3]  # Check first 3 graphs
    
# #     if len(graph_files) == 0:
# #         logger.error(f"No graphs found in {cache_path}")
# #         return
    
# #     for idx, graph_file in enumerate(graph_files):
# #         logger.info(f"\n{'='*70}")
# #         logger.info(f"Graph {idx+1}: {graph_file.name}")
# #         logger.info('='*70)
        
# #         with open(graph_file, 'rb') as f:
# #             data = pickle.load(f)
        
# #         logger.info(f"Type: {type(data)}")
# #         logger.info(f"Node types: {data.node_types}")
# #         logger.info(f"Edge types: {data.edge_types}")
        
# #         # Inspect each node type store
# #         logger.info(f"\n📦 NODE STORES:")
# #         for node_type in data.node_types:
# #             node_store = data[node_type]
# #             logger.info(f"\n  Node type: '{node_type}'")
# #             logger.info(f"  Store type: {type(node_store)}")
            
# #             # Get all keys
# #             all_keys = list(node_store.keys())
# #             logger.info(f"  Total attributes: {len(all_keys)}")
            
# #             # Separate by key type
# #             string_keys = [k for k in all_keys if isinstance(k, str)]
# #             integer_keys = [k for k in all_keys if isinstance(k, int)]
# #             other_keys = [k for k in all_keys if not isinstance(k, (str, int))]
            
# #             if string_keys:
# #                 logger.info(f"\n  ✅ String keys ({len(string_keys)}):")
# #                 for key in string_keys[:10]:  # Show first 10
# #                     value = node_store[key]
# #                     if isinstance(value, torch.Tensor):
# #                         logger.info(f"    - '{key}': Tensor {value.shape} ({value.dtype})")
# #                     else:
# #                         logger.info(f"    - '{key}': {type(value).__name__}")
            
# #             if integer_keys:
# #                 logger.info(f"\n  ⚠️ INTEGER KEYS ({len(integer_keys)}):")
# #                 for key in integer_keys[:10]:  # Show first 10
# #                     value = node_store[key]
# #                     if isinstance(value, torch.Tensor):
# #                         logger.info(f"    - {key}: Tensor {value.shape} ({value.dtype})")
# #                     else:
# #                         logger.info(f"    - {key}: {type(value).__name__} = {value}")
            
# #             if other_keys:
# #                 logger.info(f"\n  ❓ Other key types ({len(other_keys)}):")
# #                 for key in other_keys[:10]:
# #                     logger.info(f"    - {key} (type: {type(key).__name__})")
        
# #         # Inspect each edge type store
# #         logger.info(f"\n🔗 EDGE STORES:")
# #         for edge_type in data.edge_types:
# #             edge_store = data[edge_type]
# #             logger.info(f"\n  Edge type: {edge_type}")
# #             logger.info(f"  Store type: {type(edge_store)}")
            
# #             all_keys = list(edge_store.keys())
# #             logger.info(f"  Total attributes: {len(all_keys)}")
            
# #             string_keys = [k for k in all_keys if isinstance(k, str)]
# #             integer_keys = [k for k in all_keys if isinstance(k, int)]
            
# #             if string_keys:
# #                 logger.info(f"\n  ✅ String keys ({len(string_keys)}):")
# #                 for key in string_keys:
# #                     value = edge_store[key]
# #                     if isinstance(value, torch.Tensor):
# #                         logger.info(f"    - '{key}': Tensor {value.shape} ({value.dtype})")
# #                     else:
# #                         logger.info(f"    - '{key}': {type(value).__name__}")
            
# #             if integer_keys:
# #                 logger.info(f"\n  ⚠️ INTEGER KEYS ({len(integer_keys)}):")
# #                 for key in integer_keys[:10]:
# #                     value = edge_store[key]
# #                     if isinstance(value, torch.Tensor):
# #                         logger.info(f"    - {key}: Tensor {value.shape} ({value.dtype})")
# #                     else:
# #                         logger.info(f"    - {key}: {type(value).__name__} = {value}")
        
# #         # Check global attributes (non-node/edge specific)
# #         logger.info(f"\n🌐 GLOBAL ATTRIBUTES:")
# #         global_keys = [k for k in data.keys() if k not in data.node_types and k not in data.edge_types]
# #         if global_keys:
# #             for key in global_keys:
# #                 logger.info(f"  - {key} (type: {type(key).__name__})")
# #         else:
# #             logger.info("  None")
    
# #     logger.info("\n" + "="*70)
# #     logger.info("✅ Debug inspection complete!")
# #     logger.info("="*70)


# # if __name__ == '__main__':
# #     debug_graph_structure()


# """
# Verify that enriched node and edge features are correctly stored.
# """

# import pickle
# from pathlib import Path
# import torch
# import numpy as np
# import logging

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# def verify_enrichment(data_dir='data', cache_dir='cache/enriched', num_samples=10):
#     """
#     Verify node and edge enrichment in HeteroData graphs.
#     """
#     logger.info("="*70)
#     logger.info("VERIFICATION: Node & Edge Enrichment")
#     logger.info("="*70)
    
#     cache_path = Path(cache_dir) / 'train'
#     graph_files = sorted(list(cache_path.glob('*.pkl')))[:num_samples]
    
#     if len(graph_files) == 0:
#         logger.error(f"No graphs found in {cache_path}")
#         return False
    
#     logger.info(f"\nChecking {len(graph_files)} graphs...\n")
    
#     verification_passed = True
#     stats = {
#         'api_feature_dims': [],
#         'file_feature_dims': [],
#         'registry_feature_dims': [],
#         'process_feature_dims': [],
#         'network_feature_dims': [],
#         'temporal_edge_dims': [],
#         'graphs_with_temporal_features': 0,
#         'total_graphs': 0
#     }
    
#     for idx, graph_file in enumerate(graph_files):
#         logger.info(f"\n{'='*70}")
#         logger.info(f"Graph {idx+1}: {graph_file.name}")
#         logger.info('='*70)
        
#         with open(graph_file, 'rb') as f:
#             data = pickle.load(f)
        
#         stats['total_graphs'] += 1
        
#         # ============================================
#         # CHECK NODE FEATURES
#         # ============================================
#         logger.info("\n📦 NODE FEATURES:")
        
#         # Check API nodes (should be 775-dim: 768 BERT + 7 metadata)
#         if 'api' in data.node_types:
#             api_x = data['api'].x
#             api_dim = api_x.shape[1]
#             stats['api_feature_dims'].append(api_dim)
            
#             logger.info(f"\n  ✅ API nodes:")
#             logger.info(f"     Shape: {api_x.shape}")
#             logger.info(f"     Expected: [num_nodes, 775] (768 BERT + 7 metadata)")
            
#             if api_dim == 775:
#                 logger.info(f"     ✅ CORRECT: 775-dim features")
                
#                 # Check if BERT embeddings are present (first 768 dims should be normalized)
#                 bert_embeddings = api_x[:, :768]
#                 bert_mean = bert_embeddings.mean().item()
#                 bert_std = bert_embeddings.std().item()
#                 logger.info(f"     BERT embeddings (768-dim): mean={bert_mean:.4f}, std={bert_std:.4f}")
                
#                 # Check metadata features (last 7 dims)
#                 metadata = api_x[:, 768:]
#                 logger.info(f"     Metadata features (7-dim): shape={metadata.shape}")
                
#             else:
#                 logger.error(f"     ❌ WRONG DIMENSION: {api_dim} (expected 775)")
#                 verification_passed = False
        
#         # Check resource nodes (file, registry, process, network - should be 6-dim)
#         for node_type in ['file', 'registry', 'process', 'network']:
#             if node_type in data.node_types and data[node_type].num_nodes > 0:
#                 node_x = data[node_type].x
#                 node_dim = node_x.shape[1]
#                 stats[f'{node_type}_feature_dims'].append(node_dim)
                
#                 logger.info(f"\n  ✅ {node_type.upper()} nodes:")
#                 logger.info(f"     Shape: {node_x.shape}")
#                 logger.info(f"     Expected: [num_nodes, 6]")
                
#                 if node_dim == 6:
#                     logger.info(f"     ✅ CORRECT: 6-dim features")
#                 else:
#                     logger.error(f"     ❌ WRONG DIMENSION: {node_dim} (expected 6)")
#                     verification_passed = False
        
#         # ============================================
#         # CHECK EDGE FEATURES
#         # ============================================
#         logger.info("\n\n🔗 EDGE FEATURES:")
        
#         # Check temporal edges (should have 14-dim features)
#         temporal_edge_type = ('api', 'temporal', 'api')
#         if temporal_edge_type in data.edge_types:
#             edge_index = data[temporal_edge_type].edge_index
#             num_edges = edge_index.shape[1]
            
#             logger.info(f"\n  ✅ Temporal edges ('api' → 'temporal' → 'api'):")
#             logger.info(f"     Edges: {num_edges}")
            
#             if hasattr(data[temporal_edge_type], 'edge_attr'):
#                 edge_attr = data[temporal_edge_type].edge_attr
#                 edge_dim = edge_attr.shape[1]
#                 stats['temporal_edge_dims'].append(edge_dim)
#                 stats['graphs_with_temporal_features'] += 1
                
#                 logger.info(f"     Edge features shape: {edge_attr.shape}")
#                 logger.info(f"     Expected: [num_edges, 14]")
                
#                 if edge_dim == 14:
#                     logger.info(f"     ✅ CORRECT: 14-dim temporal features")
                    
#                     # Show feature statistics
#                     logger.info(f"\n     Temporal feature stats:")
#                     logger.info(f"       Mean: {edge_attr.mean(dim=0)[:5].tolist()} (first 5)")
#                     logger.info(f"       Std:  {edge_attr.std(dim=0)[:5].tolist()} (first 5)")
                    
#                     # Check for NaN/Inf
#                     if torch.isnan(edge_attr).any():
#                         logger.error(f"     ❌ Contains NaN values!")
#                         verification_passed = False
#                     if torch.isinf(edge_attr).any():
#                         logger.error(f"     ❌ Contains Inf values!")
#                         verification_passed = False
#                 else:
#                     logger.error(f"     ❌ WRONG DIMENSION: {edge_dim} (expected 14)")
#                     verification_passed = False
#             else:
#                 logger.error(f"     ❌ NO edge_attr found on temporal edges!")
#                 verification_passed = False
#         else:
#             logger.error(f"  ❌ NO temporal edges found!")
#             verification_passed = False
        
#         # Check other edge types (should have edge_index at minimum)
#         for edge_type in data.edge_types:
#             if edge_type != temporal_edge_type:
#                 edge_index = data[edge_type].edge_index
#                 num_edges = edge_index.shape[1]
#                 logger.info(f"\n  Edge type {edge_type}:")
#                 logger.info(f"     Edges: {num_edges}")
#                 logger.info(f"     Has edge_attr: {hasattr(data[edge_type], 'edge_attr')}")
    
#     # ============================================
#     # SUMMARY STATISTICS
#     # ============================================
#     logger.info("\n" + "="*70)
#     logger.info("SUMMARY STATISTICS")
#     logger.info("="*70)
    
#     logger.info(f"\nGraphs checked: {stats['total_graphs']}")
    
#     if stats['api_feature_dims']:
#         logger.info(f"\nAPI node features:")
#         logger.info(f"  Dimensions: {set(stats['api_feature_dims'])}")
#         logger.info(f"  All 775-dim: {all(d == 775 for d in stats['api_feature_dims'])}")
    
#     for node_type in ['file', 'registry', 'process', 'network']:
#         if stats[f'{node_type}_feature_dims']:
#             logger.info(f"\n{node_type.upper()} node features:")
#             logger.info(f"  Dimensions: {set(stats[f'{node_type}_feature_dims'])}")
#             logger.info(f"  All 6-dim: {all(d == 6 for d in stats[f'{node_type}_feature_dims'])}")
    
#     if stats['temporal_edge_dims']:
#         logger.info(f"\nTemporal edge features:")
#         logger.info(f"  Dimensions: {set(stats['temporal_edge_dims'])}")
#         logger.info(f"  All 14-dim: {all(d == 14 for d in stats['temporal_edge_dims'])}")
#         logger.info(f"  Graphs with temporal features: {stats['graphs_with_temporal_features']}/{stats['total_graphs']}")
    
#     logger.info("\n" + "="*70)
#     if verification_passed:
#         logger.info("✅ ALL ENRICHMENT VERIFIED - Features are intact!")
#     else:
#         logger.error("❌ ENRICHMENT ISSUES FOUND - Check errors above")
#     logger.info("="*70)
    
#     return verification_passed


# if __name__ == '__main__':
#     success = verify_enrichment(num_samples=5)  # Check first 5 graphs
#     exit(0 if success else 1)


# Quick test
import pickle
test_graph = pickle.load(open('cache/enriched/train/0a0baecb9b94a04f4ff95cda6cf027d9.pkl', 'rb'))
print(test_graph)
print(test_graph['api'])  # Check if 'timestamps' exists here
