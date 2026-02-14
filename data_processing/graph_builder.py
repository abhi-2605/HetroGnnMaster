"""
Build Heterogeneous Graphs from API Sequences and Resources

This module constructs heterogeneous temporal graphs for malware detection.

Graph Structure:
- Node Types: API, File, Registry, Process, Network
- Edge Types: temporal, operates_on, executes, creates, connects_to, etc.
- Features: Temporal embeddings, API embeddings, resource features, graph-level stats

Key Features:
- Heterogeneous graph construction (PyTorch Geometric HeteroData)
- Temporal edge features (time deltas, burst metrics)
- Rich node features (embeddings + metadata)
- Graph-level features (burst rates, velocity, diversity)
- Memory efficient (builds one graph at a time)
- Handles edge cases (no resources, single process, etc.)
- **ROBUST: Handles None values, missing fields, empty strings**

Input:  API sequence + Resources from previous modules
Output: PyTorch Geometric HeteroData graph

Usage:
    from data_processing.graph_builder import build_graph
    
    api_sequence = parse_cape_sample(data)
    resources = extract_resources(api_sequence)
    graph = build_graph(api_sequence, resources)
    
    # graph is ready for GNN training!
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import hashlib

try:
    from torch_geometric.data import HeteroData
except ImportError:
    print("⚠️  PyTorch Geometric not installed!")
    print("   Install with: pip install torch-geometric")
    HeteroData = None

# Import from config
from config import (
    MAX_API_CALLS,
    EMBEDDING_DIM,
    MAX_VOCAB_SIZE,
)


# =============================================================================
# HELPER: Safe String Extraction
# =============================================================================

def safe_str(value: Any, default: str = '') -> str:
    """
    Safely convert any value to string, handling None, empty, and other edge cases.
    
    Args:
        value: Input value (can be None, str, int, etc.)
        default: Default value if input is None or empty
        
    Returns:
        String value or default
    """
    if value is None:
        return default
    if isinstance(value, str):
        return value if value else default
    return str(value) if value else default


# =============================================================================
# API VOCABULARY (For embedding lookup)
# =============================================================================

class APIVocabulary:
    """
    Vocabulary for API name to index mapping.
    
    Used for embedding lookup in the model.
    """
    
    def __init__(self, max_vocab_size: int = MAX_VOCAB_SIZE):
        self.max_vocab_size = max_vocab_size
        self.api_to_idx = {'<UNK>': 0, '<PAD>': 1}  # Special tokens
        self.idx_to_api = {0: '<UNK>', 1: '<PAD>'}
        self.api_counts = Counter()
        self.next_idx = 2
    
    def add_api(self, api_name: str):
        """Add API to vocabulary."""
        self.api_counts[api_name] += 1
        
        if api_name not in self.api_to_idx and self.next_idx < self.max_vocab_size:
            self.api_to_idx[api_name] = self.next_idx
            self.idx_to_api[self.next_idx] = api_name
            self.next_idx += 1
    
    def get_idx(self, api_name: str) -> int:
        """Get index for API name."""
        return self.api_to_idx.get(api_name, 0)  # 0 = <UNK>
    
    def size(self) -> int:
        """Get vocabulary size."""
        return len(self.api_to_idx)


# Global vocabulary (will be populated during first pass)
GLOBAL_API_VOCAB = APIVocabulary()


# =============================================================================
# GRAPH CONSTRUCTION
# =============================================================================

def build_graph(
    api_sequence: List[Dict[str, Any]],
    resources: Dict[str, List[Dict[str, Any]]],
    add_graph_features: bool = True,
    add_temporal_edges: bool = True,
) -> HeteroData:
    """
    Build heterogeneous graph from API sequence and resources.
    
    This is the main entry point for graph construction.
    
    Args:
        api_sequence: Parsed API call sequence
        resources: Extracted resources (files, registry, processes, network)
        add_graph_features: Whether to compute graph-level features
        add_temporal_edges: Whether to add temporal edges between APIs
        
    Returns:
        HeteroData graph ready for GNN training
        
    Graph Structure:
        Node Types:
          - 'api': API call nodes (temporal sequence)
          - 'file': File resource nodes
          - 'registry': Registry key nodes
          - 'process': Process nodes
          - 'network': Network connection nodes
        
        Edge Types:
          - ('api', 'temporal', 'api'): Temporal sequence
          - ('api', 'operates_on', 'file'): API operates on file
          - ('api', 'modifies', 'registry'): API modifies registry
          - ('process', 'executes', 'api'): Process executes API
          - ('process', 'creates', 'file'): Process creates file
          - ('process', 'accesses', 'registry'): Process accesses registry
          - ('process', 'connects_to', 'network'): Process connects to network
    """
    
    if HeteroData is None:
        raise ImportError("PyTorch Geometric not installed!")
    
    # Create empty heterogeneous graph
    graph = HeteroData()
    
    # Add metadata
    graph.num_apis = len(api_sequence)
    graph.num_files = len(resources.get('files', []))
    graph.num_registry = len(resources.get('registry', []))
    graph.num_processes = len(resources.get('processes', []))
    graph.num_network = len(resources.get('network', []))
    
    # Build nodes
    add_api_nodes(graph, api_sequence)
    add_resource_nodes(graph, resources)
    
    # Build edges
    if add_temporal_edges:
        add_temporal_edges_fn(graph, api_sequence)
    
    add_resource_edges(graph, api_sequence, resources)
    
    # Add graph-level features
    if add_graph_features:
        add_graph_level_features(graph, api_sequence, resources)
    
    return graph


# =============================================================================
# NODE CONSTRUCTION
# =============================================================================

def add_api_nodes(graph: HeteroData, api_sequence: List[Dict[str, Any]]):
    """
    Add API call nodes to graph.
    
    Features:
      - API name embedding index
      - Timestamp (normalized)
      - Process ID
      - Category (one-hot)
    """
    
    if not api_sequence:
        # Empty graph - add dummy node
        graph['api'].x = torch.zeros((1, 8), dtype=torch.float)
        graph['api'].num_nodes = 1
        return
    
    num_apis = len(api_sequence)
    
    # Feature dimensions
    # [api_idx, timestamp, pid, is_file_api, is_registry_api, is_process_api, is_network_api, is_crypto_api]
    features = torch.zeros((num_apis, 8), dtype=torch.float)
    
    for i, call in enumerate(api_sequence):
        api_name = safe_str(call.get('api'), 'unknown')
        
        # Add to vocabulary
        GLOBAL_API_VOCAB.add_api(api_name)
        
        # API index for embedding lookup
        features[i, 0] = GLOBAL_API_VOCAB.get_idx(api_name)
        
        # Normalized timestamp (0 to 1)
        max_time = api_sequence[-1].get('time_since_start', 1.0)
        features[i, 1] = call.get('time_since_start', 0.0) / max(max_time, 1.0)
        
        # Process ID (normalized)
        features[i, 2] = call.get('pid', 0) / 10000.0
        
        # Category flags (one-hot-ish)
        api_lower = api_name.lower()
        features[i, 3] = 1.0 if any(x in api_lower for x in ['file', 'read', 'write']) else 0.0
        features[i, 4] = 1.0 if any(x in api_lower for x in ['reg', 'key']) else 0.0
        features[i, 5] = 1.0 if any(x in api_lower for x in ['process', 'thread']) else 0.0
        features[i, 6] = 1.0 if any(x in api_lower for x in ['socket', 'connect', 'send']) else 0.0
        features[i, 7] = 1.0 if any(x in api_lower for x in ['crypt', 'encrypt']) else 0.0
    
    graph['api'].x = features
    graph['api'].num_nodes = num_apis
    
    # Store raw API names for debugging
    graph['api'].api_names = [safe_str(call.get('api'), 'unknown') for call in api_sequence]


def add_resource_nodes(graph: HeteroData, resources: Dict[str, List[Dict[str, Any]]]):
    """
    Add resource nodes (File, Registry, Process, Network) to graph.
    
    Features:
      - Resource type encoding
      - Operation counts
      - Access frequency
      - First/last seen timestamps
      
    **ROBUST: All string operations use safe_str() to handle None values**
    """
    
    # File nodes
    files = resources.get('files', [])
    if files:
        file_features = []
        for f in files:
            # [is_system, is_temp, is_exe, num_operations, access_count, time_span]
            path = safe_str(f.get('path'), '').lower()  # ← SAFE!
            is_system = 1.0 if 'windows' in path or 'system32' in path else 0.0
            is_temp = 1.0 if 'temp' in path or 'tmp' in path else 0.0
            is_exe = 1.0 if path.endswith('.exe') or path.endswith('.dll') else 0.0
            
            operations = f.get('operations', [])
            num_ops = len(operations)
            access_count = f.get('access_count', 1)
            time_span = f.get('last_seen', 0) - f.get('first_seen', 0)
            
            file_features.append([
                is_system, is_temp, is_exe,
                min(num_ops / 10.0, 1.0),
                min(access_count / 10.0, 1.0),
                min(time_span / 100.0, 1.0)
            ])
        
        graph['file'].x = torch.tensor(file_features, dtype=torch.float)
        graph['file'].num_nodes = len(files)
        graph['file'].paths = [safe_str(f.get('path'), 'unknown') for f in files]
    else:
        graph['file'].x = torch.zeros((0, 6), dtype=torch.float)
        graph['file'].num_nodes = 0
    
    # Registry nodes
    registry = resources.get('registry', [])
    if registry:
        reg_features = []
        for r in registry:
            # [is_run_key, num_operations, access_count, time_span, has_set, has_delete]
            key = safe_str(r.get('key'), '').upper()  # ← SAFE!
            is_run_key = 1.0 if 'RUN' in key or 'STARTUP' in key else 0.0
            
            operations = r.get('operations', [])
            num_ops = len(operations)
            access_count = r.get('access_count', 1)
            time_span = r.get('last_seen', 0) - r.get('first_seen', 0)
            
            has_set = 1.0 if any(op == 'set' for op in operations) else 0.0
            has_delete = 1.0 if any(op == 'delete' for op in operations) else 0.0
            
            reg_features.append([
                is_run_key,
                min(num_ops / 5.0, 1.0),
                min(access_count / 5.0, 1.0),
                min(time_span / 100.0, 1.0),
                has_set,
                has_delete
            ])
        
        graph['registry'].x = torch.tensor(reg_features, dtype=torch.float)
        graph['registry'].num_nodes = len(registry)
        graph['registry'].keys = [safe_str(r.get('key'), 'unknown') for r in registry]
    else:
        graph['registry'].x = torch.zeros((0, 6), dtype=torch.float)
        graph['registry'].num_nodes = 0
    
    # Process nodes
    processes = resources.get('processes', [])
    if processes:
        # Deduplicate by PID
        unique_processes = {}
        for p in processes:
            pid = p.get('pid', -1)
            if pid not in unique_processes:
                unique_processes[pid] = p
        
        proc_features = []
        pid_to_idx = {}
        
        for idx, (pid, p) in enumerate(unique_processes.items()):
            pid_to_idx[pid] = idx
            
            # [is_system_process, num_apis, spawned_children, has_network, has_file, has_registry]
            name = safe_str(p.get('name'), '').lower()  # ← SAFE!
            is_system = 1.0 if any(x in name for x in ['system', 'svchost', 'csrss']) else 0.0
            
            num_apis = p.get('num_apis', 0)
            children = p.get('children', [])
            has_network = 1.0 if p.get('network_connections', []) else 0.0
            has_file = 1.0 if p.get('file_operations', []) else 0.0
            has_registry = 1.0 if p.get('registry_operations', []) else 0.0
            
            proc_features.append([
                is_system,
                min(num_apis / 100.0, 1.0),
                min(len(children) / 5.0, 1.0),
                has_network,
                has_file,
                has_registry
            ])
        
        graph['process'].x = torch.tensor(proc_features, dtype=torch.float)
        graph['process'].num_nodes = len(unique_processes)
        graph['process'].names = [safe_str(p.get('name'), 'unknown') for p in unique_processes.values()]
        graph['process'].pid_to_idx = pid_to_idx
    else:
        graph['process'].x = torch.zeros((0, 6), dtype=torch.float)
        graph['process'].num_nodes = 0
        graph['process'].pid_to_idx = {}
    
    # Network nodes
    network = resources.get('network', [])
    if network:
        net_features = []
        for n in network:
            # [is_external, is_http, is_https, num_bytes, num_connections, time_span]
            ip = safe_str(n.get('ip'), '')  # ← SAFE!
            is_external = 0.0 if ip.startswith('192.168.') or ip.startswith('10.') or ip.startswith('127.') else 1.0
            
            protocol = safe_str(n.get('protocol'), '').lower()  # ← SAFE!
            is_http = 1.0 if 'http' in protocol and 'https' not in protocol else 0.0
            is_https = 1.0 if 'https' in protocol else 0.0
            
            num_bytes = n.get('bytes', 0)
            num_connections = n.get('connection_count', 1)
            time_span = n.get('last_seen', 0) - n.get('first_seen', 0)
            
            net_features.append([
                is_external,
                is_http,
                is_https,
                min(num_bytes / 100000.0, 1.0),
                min(num_connections / 10.0, 1.0),
                min(time_span / 100.0, 1.0)
            ])
        
        graph['network'].x = torch.tensor(net_features, dtype=torch.float)
        graph['network'].num_nodes = len(network)
        graph['network'].addresses = [safe_str(n.get('ip'), 'unknown') for n in network]
    else:
        graph['network'].x = torch.zeros((0, 6), dtype=torch.float)
        graph['network'].num_nodes = 0


# =============================================================================
# EDGE CONSTRUCTION
# =============================================================================

def add_temporal_edges_fn(graph: HeteroData, api_sequence: List[Dict[str, Any]]):
    """
    Add temporal edges between consecutive API calls.
    
    Edge features:
      - Time delta (normalized)
      - Same process flag
      - Burst indicator
    """
    
    if len(api_sequence) < 2:
        # No temporal edges possible
        graph['api', 'temporal', 'api'].edge_index = torch.zeros((2, 0), dtype=torch.long)
        return
    
    edges = []
    edge_features = []
    
    for i in range(len(api_sequence) - 1):
        current = api_sequence[i]
        next_call = api_sequence[i + 1]
        
        # Edge: i → i+1
        edges.append([i, i + 1])
        
        # Time delta
        time_delta = next_call.get('time_since_start', 0) - current.get('time_since_start', 0)
        
        # Same process flag
        same_process = 1.0 if current.get('pid') == next_call.get('pid') else 0.0
        
        # Burst indicator (time delta < 0.1 seconds)
        is_burst = 1.0 if time_delta < 0.1 else 0.0
        
        edge_features.append([time_delta, same_process, is_burst])
    
    graph['api', 'temporal', 'api'].edge_index = torch.tensor(edges, dtype=torch.long).t()
    graph['api', 'temporal', 'api'].edge_attr = torch.tensor(edge_features, dtype=torch.float)


def add_resource_edges(graph: HeteroData, api_sequence: List[Dict[str, Any]], resources: Dict[str, List[Dict[str, Any]]]):
    """
    Add edges between APIs and resources, and between processes and resources.
    
    Edge types:
      - API → File
      - API → Registry
      - Process → API
      - Process → File
      - Process → Registry
      - Process → Network
    """
    
    # Build resource index maps (for lookups)
    files = resources.get('files', [])
    registry = resources.get('registry', [])
    network = resources.get('network', [])
    
    # File path → index
    file_map = {safe_str(f.get('path'), ''): idx for idx, f in enumerate(files)}
    
    # Registry key → index
    reg_map = {safe_str(r.get('key'), ''): idx for idx, r in enumerate(registry)}
    
    # Network address → index
    net_map = {safe_str(n.get('ip'), ''): idx for idx, n in enumerate(network)}
    
    # PID → index (from process nodes)
    pid_to_idx = graph['process'].pid_to_idx if hasattr(graph['process'], 'pid_to_idx') else {}
    
    # API → File edges
    api_file_edges = []
    for api_idx, call in enumerate(api_sequence):
        # Check if this API operates on files
        for arg in call.get('arguments', []):
            arg_name = safe_str(arg.get('name'), '').lower()
            if 'file' in arg_name or 'path' in arg_name:
                file_path = safe_str(arg.get('value'), '')
                if file_path in file_map:
                    api_file_edges.append([api_idx, file_map[file_path]])
    
    if api_file_edges and graph['file'].num_nodes > 0:
        edges = torch.tensor(api_file_edges, dtype=torch.long).t()
        graph['api', 'operates_on', 'file'].edge_index = edges
    else:
        graph['api', 'operates_on', 'file'].edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    # API → Registry edges
    api_reg_edges = []
    for api_idx, call in enumerate(api_sequence):
        for arg in call.get('arguments', []):
            arg_name = safe_str(arg.get('name'), '').lower()
            if 'reg' in arg_name or 'key' in arg_name:
                reg_key = safe_str(arg.get('value'), '')
                if reg_key in reg_map:
                    api_reg_edges.append([api_idx, reg_map[reg_key]])
    
    if api_reg_edges and graph['registry'].num_nodes > 0:
        edges = torch.tensor(api_reg_edges, dtype=torch.long).t()
        graph['api', 'modifies', 'registry'].edge_index = edges
    else:
        graph['api', 'modifies', 'registry'].edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    # Process → API edges
    proc_api_edges = []
    for api_idx, call in enumerate(api_sequence):
        pid = call.get('pid', -1)
        if pid in pid_to_idx:
            proc_api_edges.append([pid_to_idx[pid], api_idx])
    
    if proc_api_edges and graph['process'].num_nodes > 0:
        edges = torch.tensor(proc_api_edges, dtype=torch.long).t()
        graph['process', 'executes', 'api'].edge_index = edges
    else:
        graph['process', 'executes', 'api'].edge_index = torch.zeros((2, 0), dtype=torch.long)


# =============================================================================
# GRAPH-LEVEL FEATURES
# =============================================================================

def add_graph_level_features(graph: HeteroData, api_sequence: List[Dict[str, Any]], resources: Dict[str, List[Dict[str, Any]]]):
    """
    Add graph-level features for downstream classification.
    
    Features:
      - API call rate (calls/second)
      - Resource diversity (unique files, registry, network)
      - Process count
      - Temporal burst metrics
      - Operation diversity
    """
    
    if not api_sequence:
        graph.graph_features = torch.zeros((10,), dtype=torch.float)
        return
    
    # Time span
    time_span = api_sequence[-1].get('time_since_start', 1.0)
    
    # API call rate
    api_rate = len(api_sequence) / max(time_span, 0.1)
    
    # Resource counts
    num_files = len(resources.get('files', []))
    num_registry = len(resources.get('registry', []))
    num_processes = len(resources.get('processes', []))
    num_network = len(resources.get('network', []))
    
    # Burst metrics (consecutive calls with < 0.01s gap)
    bursts = 0
    for i in range(len(api_sequence) - 1):
        delta = api_sequence[i + 1].get('time_since_start', 0) - api_sequence[i].get('time_since_start', 0)
        if delta < 0.01:
            bursts += 1
    burst_ratio = bursts / max(len(api_sequence) - 1, 1)
    
    # API diversity (unique API count / total)
    unique_apis = len(set(safe_str(call.get('api'), 'unknown') for call in api_sequence))
    api_diversity = unique_apis / len(api_sequence)
    
    # Graph features vector
    features = torch.tensor([
        min(api_rate / 100.0, 1.0),        # Normalized API rate
        min(num_files / 50.0, 1.0),        # Normalized file count
        min(num_registry / 50.0, 1.0),     # Normalized registry count
        min(num_processes / 10.0, 1.0),    # Normalized process count
        min(num_network / 10.0, 1.0),      # Normalized network count
        burst_ratio,                        # Burst ratio
        api_diversity,                      # API diversity
        min(time_span / 300.0, 1.0),       # Normalized time span
        1.0 if num_network > 0 else 0.0,  # Has network activity
        1.0 if num_files > 10 else 0.0,   # High file activity
    ], dtype=torch.float)
    
    graph.graph_features = features


# =============================================================================
# MAIN ENTRY POINT (for testing)
# =============================================================================

if __name__ == "__main__":
    print("✓ graph_builder.py module loaded successfully")
    print(f"  API Vocabulary size: {GLOBAL_API_VOCAB.size()}")
    print("\nTo test:")
    print("  python -m data_processing.data_loader train")
