"""
Extract Resource Nodes from API Sequences

This module extracts and structures resource nodes (Files, Registry, Processes, Network)
from parsed API call sequences for heterogeneous graph construction.

Key Features:
- Extracts 4 resource types: Files, Registry, Processes, Network
- Deduplicates resources (same file accessed 100x → 1 node)
- Tracks operations per resource (create, read, write, delete)
- Extracts rich metadata (paths, IPs, ports, domains, keys)
- Maps APIs to resource types automatically
- Memory efficient (processes one sample at a time)

Input:  Parsed API sequence from parse_cape_json.py
Output: Structured resource dictionaries ready for graph construction

Usage:
    from data_processing.extract_resources import extract_resources
    
    api_sequence = parse_cape_sample(data)
    resources = extract_resources(api_sequence)
    
    # resources = {
    #     'files': [...],
    #     'registry': [...],
    #     'processes': [...],
    #     'network': [...],
    # }
"""

from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from collections import defaultdict, Counter
import re

# Import from config
from config import (
    MAX_PATH_LENGTH,
    MAX_REGISTRY_KEY_LENGTH,
)


# =============================================================================
# API TO RESOURCE TYPE MAPPING
# =============================================================================

# File-related APIs
FILE_APIS = {
    # File operations
    'NtCreateFile', 'NtOpenFile', 'NtReadFile', 'NtWriteFile',
    'CreateFileW', 'CreateFileA', 'ReadFile', 'WriteFile',
    'DeleteFileW', 'DeleteFileA', 'NtDeleteFile',
    'MoveFileWithProgressW', 'CopyFileW', 'CopyFileExW',
    'SetFileAttributesW', 'GetFileAttributesW',
    'FindFirstFileExW', 'FindNextFileW',
    
    # Directory operations
    'NtCreateDirectoryObject', 'CreateDirectoryW', 'RemoveDirectoryW',
    'NtQueryDirectoryFile',
}

# Registry-related APIs
REGISTRY_APIS = {
    'NtCreateKey', 'NtOpenKey', 'NtOpenKeyEx',
    'NtSetValueKey', 'NtQueryValueKey', 'NtDeleteValueKey',
    'NtDeleteKey', 'NtEnumerateKey', 'NtEnumerateValueKey',
    'RegOpenKeyExA', 'RegOpenKeyExW',
    'RegCreateKeyExA', 'RegCreateKeyExW',
    'RegSetValueExA', 'RegSetValueExW',
    'RegQueryValueExA', 'RegQueryValueExW',
    'RegDeleteKeyA', 'RegDeleteKeyW',
    'RegDeleteValueA', 'RegDeleteValueW',
}

# Process-related APIs
PROCESS_APIS = {
    'NtCreateProcess', 'NtCreateProcessEx', 'NtCreateUserProcess',
    'CreateProcessInternalW', 'CreateProcessW', 'CreateProcessA',
    'NtTerminateProcess', 'TerminateProcess',
    'NtOpenProcess', 'OpenProcess',
    'NtResumeThread', 'NtSuspendThread',
    'CreateRemoteThread', 'NtCreateThreadEx',
    'ShellExecuteExW', 'WinExec',
}

# Network-related APIs
NETWORK_APIS = {
    # Socket operations
    'connect', 'send', 'recv', 'WSASend', 'WSARecv',
    'socket', 'bind', 'listen', 'accept',
    'WSAConnect', 'WSASocketW',
    'closesocket', 'shutdown',
    
    # High-level network
    'InternetConnectA', 'InternetConnectW',
    'InternetOpenUrlA', 'InternetOpenUrlW',
    'HttpOpenRequestA', 'HttpOpenRequestW',
    'HttpSendRequestA', 'HttpSendRequestW',
    'InternetReadFile', 'InternetWriteFile',
    
    # DNS
    'gethostbyname', 'getaddrinfo', 'GetAddrInfoW',
    'DnsQuery_A', 'DnsQuery_W',
}


# =============================================================================
# RESOURCE EXTRACTION FUNCTIONS
# =============================================================================

def extract_resources(
    api_sequence: List[Dict[str, Any]],
    deduplicate: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract all resource nodes from an API sequence.
    
    This is the main entry point that coordinates extraction of all resource types.
    
    Args:
        api_sequence: Parsed API call sequence from parse_cape_json
        deduplicate: Whether to deduplicate resources (recommended: True)
        
    Returns:
        Dictionary with 4 resource types:
        {
            'files': [list of file resources],
            'registry': [list of registry resources],
            'processes': [list of process resources],
            'network': [list of network resources],
        }
        
    Example:
        >>> resources = extract_resources(api_sequence)
        >>> len(resources['files'])
        45
        >>> resources['files'][0]
        {
            'path': 'C:\\Users\\doc.txt',
            'operations': ['create', 'write', 'delete'],
            'api_indices': [0, 5, 12],
            'first_seen': 0.0,
            'last_seen': 2.5,
        }
    """
    
    if not api_sequence:
        return {
            'files': [],
            'registry': [],
            'processes': [],
            'network': [],
        }
    
    # Extract each resource type
    files = extract_file_resources(api_sequence)
    registry = extract_registry_resources(api_sequence)
    processes = extract_process_resources(api_sequence)
    network = extract_network_resources(api_sequence)
    
    # Deduplicate if requested
    if deduplicate:
        files = deduplicate_resources(files, key='path')
        registry = deduplicate_resources(registry, key='key')
        processes = deduplicate_resources(processes, key='pid')
        network = deduplicate_resources(network, key='address')
    
    return {
        'files': files,
        'registry': registry,
        'processes': processes,
        'network': network,
    }


# =============================================================================
# FILE RESOURCE EXTRACTION
# =============================================================================

def extract_file_resources(
    api_sequence: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Extract file resources from API sequence.
    
    Looks for file-related APIs and extracts file paths from arguments.
    
    Returns:
        List of file resource dictionaries
    """
    
    file_resources = []
    
    for idx, call in enumerate(api_sequence):
        api_name = call.get('api', '')
        
        # Check if this is a file API
        if api_name not in FILE_APIS:
            continue
        
        # Extract file path from arguments
        file_path = extract_file_path(call.get('arguments', []))
        
        if not file_path:
            continue
        
        # Normalize path
        file_path = normalize_path(file_path)
        
        # Skip if too long (likely corrupted)
        if len(file_path) > MAX_PATH_LENGTH:
            continue
        
        # Determine operation type
        operation = categorize_file_operation(api_name)
        
        # Create resource record
        resource = {
            'path': file_path,
            'operation': operation,
            'api': api_name,
            'api_index': call.get('call_index', idx),
            'timestamp': call.get('time_since_start', 0.0),
            'process_name': call.get('process_name', 'unknown'),
            'pid': call.get('pid', -1),
        }
        
        file_resources.append(resource)
    
    return file_resources


def extract_file_path(arguments: List[Dict[str, Any]]) -> Optional[str]:
    """
    Extract file path from API call arguments.
    
    Searches for common argument names that contain file paths.
    
    Args:
        arguments: List of argument dictionaries
        
    Returns:
        Extracted file path or None
    """
    
    if not arguments:
        return None
    
    # Common argument names for file paths
    path_keys = [
        'FilePath', 'FileName', 'Path', 'lpFileName',
        'ObjectAttributes', 'NewFilePath', 'ExistingFileName',
        'Buffer', 'DirectoryPath', 'file_path', 'filepath',
    ]
    
    # Convert list of dicts to dict (if needed)
    if isinstance(arguments, list):
        arg_dict = {}
        for arg in arguments:
            if isinstance(arg, dict):
                name = arg.get('name', arg.get('argument_name'))
                value = arg.get('value', arg.get('argument_value'))
                if name:
                    arg_dict[name] = value
    else:
        arg_dict = arguments
    
    # Search for file path
    for key in path_keys:
        if key in arg_dict:
            value = arg_dict[key]
            if value and isinstance(value, str) and len(value) > 0:
                # Check if it looks like a path
                if ('\\' in value or '/' in value or 
                    value.startswith('C:') or value.startswith('\\\\')):
                    return value
    
    return None


def categorize_file_operation(api_name: str) -> str:
    """
    Categorize file API into operation type.
    
    Args:
        api_name: API function name
        
    Returns:
        Operation category: 'create', 'read', 'write', 'delete', 'modify', 'query'
    """
    
    api_lower = api_name.lower()
    
    if 'create' in api_lower:
        return 'create'
    elif 'write' in api_lower or 'set' in api_lower:
        return 'write'
    elif 'read' in api_lower or 'query' in api_lower or 'get' in api_lower:
        return 'read'
    elif 'delete' in api_lower or 'remove' in api_lower:
        return 'delete'
    elif 'move' in api_lower or 'copy' in api_lower or 'rename' in api_lower:
        return 'modify'
    elif 'open' in api_lower:
        return 'open'
    elif 'find' in api_lower or 'enumerate' in api_lower:
        return 'query'
    else:
        return 'other'


# =============================================================================
# REGISTRY RESOURCE EXTRACTION
# =============================================================================

def extract_registry_resources(
    api_sequence: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Extract registry resources from API sequence.
    
    Looks for registry-related APIs and extracts registry keys.
    
    Returns:
        List of registry resource dictionaries
    """
    
    registry_resources = []
    
    for idx, call in enumerate(api_sequence):
        api_name = call.get('api', '')
        
        # Check if this is a registry API
        if api_name not in REGISTRY_APIS:
            continue
        
        # Extract registry key from arguments
        reg_key = extract_registry_key(call.get('arguments', []))
        
        if not reg_key:
            continue
        
        # Normalize key
        reg_key = normalize_registry_key(reg_key)
        
        # Skip if too long
        if len(reg_key) > MAX_REGISTRY_KEY_LENGTH:
            continue
        
        # Determine operation type
        operation = categorize_registry_operation(api_name)
        
        # Create resource record
        resource = {
            'key': reg_key,
            'operation': operation,
            'api': api_name,
            'api_index': call.get('call_index', idx),
            'timestamp': call.get('time_since_start', 0.0),
            'process_name': call.get('process_name', 'unknown'),
            'pid': call.get('pid', -1),
        }
        
        registry_resources.append(resource)
    
    return registry_resources


def extract_registry_key(arguments: List[Dict[str, Any]]) -> Optional[str]:
    """
    Extract registry key from API call arguments.
    
    Args:
        arguments: List of argument dictionaries
        
    Returns:
        Extracted registry key or None
    """
    
    if not arguments:
        return None
    
    # Common argument names for registry keys
    key_names = [
        'KeyName', 'RegistryKey', 'Key', 'FullName',
        'SubKey', 'lpSubKey', 'hKey', 'Handle',
    ]
    
    # Convert list to dict
    if isinstance(arguments, list):
        arg_dict = {}
        for arg in arguments:
            if isinstance(arg, dict):
                name = arg.get('name', arg.get('argument_name'))
                value = arg.get('value', arg.get('argument_value'))
                if name:
                    arg_dict[name] = value
    else:
        arg_dict = arguments
    
    # Search for registry key
    for key in key_names:
        if key in arg_dict:
            value = arg_dict[key]
            if value and isinstance(value, str):
                # Check if it looks like a registry key
                if any(x in value.upper() for x in ['HKEY', 'HKLM', 'HKCU', 'HKCR', 'HKU', 'HKCC']):
                    return value
    
    return None


def categorize_registry_operation(api_name: str) -> str:
    """
    Categorize registry API into operation type.
    
    Args:
        api_name: API function name
        
    Returns:
        Operation category: 'create', 'open', 'set', 'query', 'delete', 'enumerate'
    """
    
    api_lower = api_name.lower()
    
    if 'create' in api_lower:
        return 'create'
    elif 'open' in api_lower:
        return 'open'
    elif 'set' in api_lower or 'write' in api_lower:
        return 'set'
    elif 'query' in api_lower or 'get' in api_lower:
        return 'query'
    elif 'delete' in api_lower:
        return 'delete'
    elif 'enumerate' in api_lower:
        return 'enumerate'
    else:
        return 'other'


# =============================================================================
# PROCESS RESOURCE EXTRACTION
# =============================================================================

def extract_process_resources(
    api_sequence: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Extract process resources from API sequence.
    
    Extracts both:
    1. Processes from API calls (CreateProcess, etc.)
    2. Processes that made the API calls (from call context)
    
    Returns:
        List of process resource dictionaries
    """
    
    process_resources = []
    seen_pids = set()
    
    # Extract processes from call context (all processes that executed APIs)
    for idx, call in enumerate(api_sequence):
        pid = call.get('pid', -1)
        
        if pid == -1 or pid in seen_pids:
            continue
        
        seen_pids.add(pid)
        
        resource = {
            'pid': pid,
            'name': call.get('process_name', 'unknown'),
            'ppid': call.get('ppid', -1),
            'first_seen': call.get('time_since_start', 0.0),
            'api_count': 0,  # Will be updated later
            'source': 'context',
        }
        
        process_resources.append(resource)
    
    # Extract processes created by APIs (CreateProcess, etc.)
    for idx, call in enumerate(api_sequence):
        api_name = call.get('api', '')
        
        if api_name not in PROCESS_APIS:
            continue
        
        # Extract process name/path from arguments
        process_info = extract_process_info(call.get('arguments', []))
        
        if not process_info:
            continue
        
        resource = {
            'name': process_info.get('name', 'unknown'),
            'path': process_info.get('path'),
            'command_line': process_info.get('command_line'),
            'api': api_name,
            'api_index': call.get('call_index', idx),
            'timestamp': call.get('time_since_start', 0.0),
            'creator_pid': call.get('pid', -1),
            'source': 'created',
        }
        
        process_resources.append(resource)
    
    return process_resources


def extract_process_info(arguments: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Extract process information from API arguments.
    
    Args:
        arguments: List of argument dictionaries
        
    Returns:
        Dictionary with process info or None
    """
    
    if not arguments:
        return None
    
    # Convert to dict
    if isinstance(arguments, list):
        arg_dict = {}
        for arg in arguments:
            if isinstance(arg, dict):
                name = arg.get('name', arg.get('argument_name'))
                value = arg.get('value', arg.get('argument_value'))
                if name:
                    arg_dict[name] = value
    else:
        arg_dict = arguments
    
    info = {}
    
    # Extract process name/path
    path_keys = ['ApplicationName', 'lpApplicationName', 'FilePath', 'ProcessName']
    for key in path_keys:
        if key in arg_dict and arg_dict[key]:
            info['path'] = arg_dict[key]
            info['name'] = Path(arg_dict[key]).name
            break
    
    # Extract command line
    cmd_keys = ['CommandLine', 'lpCommandLine', 'Parameters']
    for key in cmd_keys:
        if key in arg_dict and arg_dict[key]:
            info['command_line'] = arg_dict[key]
            break
    
    return info if info else None


# =============================================================================
# NETWORK RESOURCE EXTRACTION
# =============================================================================

def extract_network_resources(
    api_sequence: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Extract network resources from API sequence.
    
    Extracts IPs, domains, ports from network-related APIs.
    
    Returns:
        List of network resource dictionaries
    """
    
    network_resources = []
    
    for idx, call in enumerate(api_sequence):
        api_name = call.get('api', '')
        
        # Check if this is a network API
        if api_name not in NETWORK_APIS:
            continue
        
        # Extract network information
        net_info = extract_network_info(call.get('arguments', []))
        
        if not net_info:
            continue
        
        # Determine operation type
        operation = categorize_network_operation(api_name)
        
        # Create resource record
        resource = {
            'address': net_info.get('address', 'unknown'),
            'ip': net_info.get('ip'),
            'domain': net_info.get('domain'),
            'port': net_info.get('port'),
            'protocol': net_info.get('protocol', 'unknown'),
            'operation': operation,
            'api': api_name,
            'api_index': call.get('call_index', idx),
            'timestamp': call.get('time_since_start', 0.0),
            'process_name': call.get('process_name', 'unknown'),
            'pid': call.get('pid', -1),
        }
        
        network_resources.append(resource)
    
    return network_resources


def extract_network_info(arguments: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Extract network information from API arguments.
    
    Args:
        arguments: List of argument dictionaries
        
    Returns:
        Dictionary with network info or None
    """
    
    if not arguments:
        return None
    
    # Convert to dict
    if isinstance(arguments, list):
        arg_dict = {}
        for arg in arguments:
            if isinstance(arg, dict):
                name = arg.get('name', arg.get('argument_name'))
                value = arg.get('value', arg.get('argument_value'))
                if name:
                    arg_dict[name] = value
    else:
        arg_dict = arguments
    
    info = {}
    
    # Extract IP address
    ip_keys = ['IPAddress', 'ip_address', 'RemoteAddress', 'ServerName']
    for key in ip_keys:
        if key in arg_dict and arg_dict[key]:
            value = str(arg_dict[key])
            if is_ip_address(value):
                info['ip'] = value
                info['address'] = value
            elif is_domain(value):
                info['domain'] = value
                info['address'] = value
    
    # Extract port
    port_keys = ['Port', 'port', 'RemotePort', 'ServerPort']
    for key in port_keys:
        if key in arg_dict and arg_dict[key]:
            try:
                info['port'] = int(arg_dict[key])
            except:
                pass
    
    # Extract domain/URL
    url_keys = ['URL', 'url', 'HostName', 'hostname', 'ServerName']
    for key in url_keys:
        if key in arg_dict and arg_dict[key]:
            value = str(arg_dict[key])
            if is_domain(value) or 'http' in value.lower():
                info['domain'] = value
                if 'address' not in info:
                    info['address'] = value
    
    # Combine IP:Port for address
    if 'ip' in info and 'port' in info:
        info['address'] = f"{info['ip']}:{info['port']}"
    
    return info if info else None


def categorize_network_operation(api_name: str) -> str:
    """
    Categorize network API into operation type.
    
    Args:
        api_name: API function name
        
    Returns:
        Operation category
    """
    
    api_lower = api_name.lower()
    
    if 'connect' in api_lower:
        return 'connect'
    elif 'send' in api_lower or 'write' in api_lower:
        return 'send'
    elif 'recv' in api_lower or 'read' in api_lower:
        return 'receive'
    elif 'dns' in api_lower or 'gethostbyname' in api_lower or 'getaddrinfo' in api_lower:
        return 'dns_query'
    elif 'socket' in api_lower:
        return 'create_socket'
    elif 'bind' in api_lower:
        return 'bind'
    elif 'listen' in api_lower:
        return 'listen'
    elif 'accept' in api_lower:
        return 'accept'
    elif 'close' in api_lower or 'shutdown' in api_lower:
        return 'close'
    else:
        return 'other'


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def normalize_path(path: str) -> str:
    """
    Normalize file path for consistency.
    
    - Convert to lowercase
    - Replace forward slashes with backslashes
    - Remove duplicate backslashes
    - Strip whitespace
    
    Args:
        path: Raw file path
        
    Returns:
        Normalized path
    """
    
    if not path:
        return ''
    
    # Convert to lowercase
    path = path.lower()
    
    # Replace forward slashes
    path = path.replace('/', '\\')
    
    # Remove duplicate backslashes
    path = re.sub(r'\\+', r'\\', path)
    
    # Strip whitespace
    path = path.strip()
    
    return path


def normalize_registry_key(key: str) -> str:
    """
    Normalize registry key for consistency.
    
    - Convert to uppercase
    - Standardize hive names (HKLM, HKCU, etc.)
    - Strip whitespace
    
    Args:
        key: Raw registry key
        
    Returns:
        Normalized key
    """
    
    if not key:
        return ''
    
    # Convert to uppercase
    key = key.upper()
    
    # Standardize hive names
    replacements = {
        'HKEY_LOCAL_MACHINE': 'HKLM',
        'HKEY_CURRENT_USER': 'HKCU',
        'HKEY_CLASSES_ROOT': 'HKCR',
        'HKEY_USERS': 'HKU',
        'HKEY_CURRENT_CONFIG': 'HKCC',
    }
    
    for old, new in replacements.items():
        if key.startswith(old):
            key = key.replace(old, new, 1)
            break
    
    # Strip whitespace
    key = key.strip()
    
    return key


def is_ip_address(value: str) -> bool:
    """
    Check if a string is an IP address.
    
    Args:
        value: String to check
        
    Returns:
        True if IP address, False otherwise
    """
    
    # Simple regex for IPv4
    pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    
    if not re.match(pattern, value):
        return False
    
    # Validate octets
    parts = value.split('.')
    return all(0 <= int(part) <= 255 for part in parts)


def is_domain(value: str) -> bool:
    """
    Check if a string is a domain name.
    
    Args:
        value: String to check
        
    Returns:
        True if domain, False otherwise
    """
    
    # Simple domain pattern
    pattern = r'^[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(\.[a-zA-Z0-9]([a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+$'
    
    return bool(re.match(pattern, value))


def deduplicate_resources(
    resources: List[Dict[str, Any]],
    key: str,
) -> List[Dict[str, Any]]:
    """
    Deduplicate resources by key and merge operations.
    
    Example:
        Input: [
            {'path': 'C:\\doc.txt', 'operation': 'create'},
            {'path': 'C:\\doc.txt', 'operation': 'write'},
            {'path': 'C:\\doc.txt', 'operation': 'delete'},
        ]
        
        Output: [
            {
                'path': 'C:\\doc.txt',
                'operations': ['create', 'write', 'delete'],
                'api_indices': [0, 5, 12],
                'first_seen': 0.0,
                'last_seen': 2.5,
                'access_count': 3,
            }
        ]
    
    Args:
        resources: List of resource dictionaries
        key: Key to deduplicate on (e.g., 'path', 'key', 'pid')
        
    Returns:
        Deduplicated list of resources
    """
    
    if not resources:
        return []
    
    # Group by key
    grouped = defaultdict(list)
    
    for resource in resources:
        key_value = resource.get(key)
        if key_value is not None:
            grouped[key_value].append(resource)
    
    # Merge grouped resources
    deduplicated = []
    
    for key_value, group in grouped.items():
        # Collect all operations
        operations = []
        api_indices = []
        timestamps = []
        apis = []
        
        for item in group:
            op = item.get('operation')
            if op and op not in operations:
                operations.append(op)
            
            api_idx = item.get('api_index')
            if api_idx is not None:
                api_indices.append(api_idx)
            
            ts = item.get('timestamp')
            if ts is not None:
                timestamps.append(ts)
            
            api = item.get('api')
            if api and api not in apis:
                apis.append(api)
        
        # Create merged resource
        merged = group[0].copy()  # Start with first item
        
        merged['operations'] = operations
        merged['api_indices'] = sorted(api_indices) if api_indices else []
        merged['first_seen'] = min(timestamps) if timestamps else 0.0
        merged['last_seen'] = max(timestamps) if timestamps else 0.0
        merged['access_count'] = len(group)
        merged['apis'] = apis
        
        # Remove single 'operation' and 'api_index' (replaced by plurals)
        merged.pop('operation', None)
        merged.pop('api_index', None)
        merged.pop('api', None)
        merged.pop('timestamp', None)
        
        deduplicated.append(merged)
    
    return deduplicated


# =============================================================================
# STATISTICS AND DEBUGGING
# =============================================================================

def summarize_resources(resources: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Generate statistics about extracted resources.
    
    Args:
        resources: Dictionary of resource lists
        
    Returns:
        Statistics dictionary
    """
    
    stats = {
        'num_files': len(resources.get('files', [])),
        'num_registry': len(resources.get('registry', [])),
        'num_processes': len(resources.get('processes', [])),
        'num_network': len(resources.get('network', [])),
        'total_resources': sum(len(v) for v in resources.values()),
    }
    
    # File statistics
    if resources.get('files'):
        file_ops = [op for f in resources['files'] 
                    for op in f.get('operations', [])]
        stats['file_operations'] = dict(Counter(file_ops))
    
    # Registry statistics
    if resources.get('registry'):
        reg_ops = [op for r in resources['registry'] 
                   for op in r.get('operations', [])]
        stats['registry_operations'] = dict(Counter(reg_ops))
    
    # Network statistics
    if resources.get('network'):
        net_ops = [op for n in resources['network'] 
                   for op in n.get('operations', [])]
        stats['network_operations'] = dict(Counter(net_ops))
    
    return stats


# =============================================================================
# TESTING
# =============================================================================

def test_extract_on_sample(sample_file: str) -> bool:
    """
    Test resource extraction on a single sample.
    
    Args:
        sample_file: Path to JSON file
        
    Returns:
        True if successful, False otherwise
    """
    
    print("=" * 70)
    print("TESTING RESOURCE EXTRACTION")
    print("=" * 70)
    
    sample_path = Path(sample_file)
    
    if not sample_path.exists():
        print(f"❌ File not found: {sample_file}")
        return False
    
    print(f"\n📄 Testing on: {sample_path.name}")
    
    # Parse API sequence
    print("\n🔍 Parsing API sequence...")
    api_sequence = load_and_parse_json_file(sample_path)
    
    if not api_sequence:
        print("⚠️  Failed to parse API sequence!")
        print("   Reason: File is empty, malformed, or has no valid API calls")
        return False
    
    print(f"✓ Parsed {len(api_sequence)} API calls")
    
    # Extract resources
    print("\n🔍 Extracting resources...")
    resources = extract_resources(api_sequence)
    
    # Print summary
    stats = summarize_resources(resources)
    
    print(f"\n📊 EXTRACTION RESULTS:")
    print(f"  Files:     {stats['num_files']}")
    print(f"  Registry:  {stats['num_registry']}")
    print(f"  Processes: {stats['num_processes']}")
    print(f"  Network:   {stats['num_network']}")
    print(f"  ──────────────────")
    print(f"  Total:     {stats['total_resources']}")
    
    if stats.get('file_operations'):
        print(f"\n📂 FILE OPERATIONS:")
        for op, count in stats['file_operations'].items():
            print(f"  {op:12} : {count:3}")
    
    if stats.get('registry_operations'):
        print(f"\n📋 REGISTRY OPERATIONS:")
        for op, count in stats['registry_operations'].items():
            print(f"  {op:12} : {count:3}")
    
    if stats.get('network_operations'):
        print(f"\n🌐 NETWORK OPERATIONS:")
        for op, count in stats['network_operations'].items():
            print(f"  {op:12} : {count:3}")
    
    # Show examples
    print(f"\n📋 EXAMPLE RESOURCES:")
    
    if resources['files']:
        print(f"\n  Files (showing first 3):")
        for f in resources['files'][:3]:
            print(f"    - {f.get('path', 'unknown')[:60]}")
            print(f"      Operations: {f.get('operations', [])}")
            print(f"      Access count: {f.get('access_count', 1)}")
    
    if resources['registry']:
        print(f"\n  Registry (showing first 3):")
        for r in resources['registry'][:3]:
            print(f"    - {r.get('key', 'unknown')[:60]}")
            print(f"      Operations: {r.get('operations', [])}")
    
    if resources['network']:
        print(f"\n  Network (showing first 3):")
        for n in resources['network'][:3]:
            print(f"    - {n.get('address', 'unknown')}")
            print(f"      Protocol: {n.get('protocol', 'unknown')}")
            print(f"      Operations: {n.get('operations', [])}")
    
    print("\n" + "=" * 70)
    return True


if __name__ == "__main__":
    """
    Test resource extraction on sample files.
    
    This will try multiple files until it finds a valid one.
    
    Usage:
        # Test on first valid file found
        python -m data_processing.extract_resources
        
        # Test on specific file
        python -m data_processing.extract_resources path/to/file.json
    """
    
    import sys
    from .parse_cape_json import load_and_parse_json_file
    
    # User provided a specific file
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        success = test_extract_on_sample(test_file)
        sys.exit(0 if success else 1)
    
    # Auto-find valid sample files
    from config import RAW_SAMPLE_DIR
    
    print("🔍 Searching for valid JSON files in sample directory...")
    sample_dir = Path(RAW_SAMPLE_DIR)
    
    json_files = list(sample_dir.rglob('*.json'))
    
    if not json_files:
        print("❌ No JSON files found in data/sample/")
        sys.exit(1)
    
    print(f"✓ Found {len(json_files)} JSON files")
    print(f"🔄 Testing files until we find a valid one...\n")
    
    # Try up to 10 files
    max_attempts = min(10, len(json_files))
    
    for i, json_file in enumerate(json_files[:max_attempts]):
        print(f"\n{'─' * 70}")
        print(f"ATTEMPT {i + 1}/{max_attempts}")
        print(f"{'─' * 70}")
        
        success = test_extract_on_sample(str(json_file))
        
        if success:
            print("\n" + "=" * 70)
            print("✅ TESTING COMPLETE - Found valid sample!")
            print("=" * 70)
            print("\n💡 To test a specific file, run:")
            print(f"   python -m data_processing.extract_resources path/to/file.json")
            sys.exit(0)
        else:
            print(f"   → Skipping this file, trying next one...\n")
    
    # If we get here, no valid files found
    print("\n" + "=" * 70)
    print("⚠️  NO VALID FILES FOUND")
    print("=" * 70)
    print(f"\nTried {max_attempts} files, all were empty or malformed.")
    print("\nCheck your data/sample/ directory!")
    sys.exit(1)

