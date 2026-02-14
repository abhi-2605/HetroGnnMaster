"""
Parse Cuckoo/CAPE Sandbox JSON Reports

This module extracts temporal API call sequences from sandbox execution traces.

Key Features:
- Extracts API calls from all processes
- Creates unified temporal sequence (sorted by timestamp)
- Normalizes timestamps (relative to execution start)
- Handles empty/malformed JSON files gracefully
- Memory efficient (processes one file at a time)
- Extracts rich context (process, arguments, return values)

Input:  Complex nested JSON from Cuckoo/CAPE sandbox
Output: Clean temporal API sequence ready for graph construction

Usage:
    from data_processing.parse_cape_json import parse_cape_sample
    
    with open('sample.json') as f:
        data = json.load(f)
    
    api_sequence = parse_cape_sample(data)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import re

# Import from config
from config import (
    MAX_API_CALLS,
    MIN_API_CALLS,
)


# =============================================================================
# API FILTERING (Optional - Remove noisy/benign APIs)
# =============================================================================

# Common benign APIs that add noise (can be filtered out)
BENIGN_APIS = {
    # Threading/synchronization (very common, less informative)
    'NtWaitForSingleObject',
    'NtWaitForMultipleObjects',
    'NtDelayExecution',
    'NtQueryPerformanceCounter',
    
    # Memory management (too common)
    'NtAllocateVirtualMemory',
    'NtFreeVirtualMemory',
    'NtProtectVirtualMemory',
    
    # Process/thread queries (benign information gathering)
    'NtQueryInformationProcess',
    'NtQueryInformationThread',
    'NtQuerySystemInformation',
}

# Set to True to filter benign APIs, False to keep all
FILTER_BENIGN_APIS = False  # Start with False, can enable later for optimization


# =============================================================================
# CORE PARSING FUNCTIONS
# =============================================================================
from datetime import datetime

def parse_timestamp(timestamp: Any) -> float:
    """
    Parse timestamp from various formats.
    
    Supports:
    - Unix timestamp (float): 1234567890.123
    - Datetime string: '2022-02-14 14:41:31,276'
    - ISO format: '2022-02-14T14:41:31.276'
    
    Args:
        timestamp: Raw timestamp value
        
    Returns:
        Unix timestamp as float
    """
    
    if timestamp is None:
        return 0.0
    
    # Already a number (Unix timestamp)
    if isinstance(timestamp, (int, float)):
        return float(timestamp)
    
    # String - need to parse
    if isinstance(timestamp, str):
        try:
            # Try parsing as float first
            return float(timestamp)
        except ValueError:
            pass
        
        try:
            # Handle format: '2022-02-14 14:41:31,276'
            # Replace comma with dot for milliseconds
            timestamp_clean = timestamp.replace(',', '.')
            
            # Try parsing with milliseconds
            if '.' in timestamp_clean:
                dt = datetime.strptime(timestamp_clean, '%Y-%m-%d %H:%M:%S.%f')
            else:
                dt = datetime.strptime(timestamp_clean, '%Y-%m-%d %H:%M:%S')
            
            # Convert to Unix timestamp
            return dt.timestamp()
            
        except ValueError:
            pass
        
        try:
            # Try ISO format: '2022-02-14T14:41:31.276'
            dt = datetime.fromisoformat(timestamp.replace(',', '.'))
            return dt.timestamp()
        except:
            pass
    
    # Fallback: return 0
    return 0.0

def parse_cape_sample(
    json_ : Dict[str, Any],
    max_calls: int = MAX_API_CALLS,
    min_calls: int = MIN_API_CALLS,
    filter_benign: bool = FILTER_BENIGN_APIS,
) -> Optional[List[Dict[str, Any]]]:
    """
    Parse a Cuckoo/CAPE JSON report and extract temporal API sequence.
    
    This function:
    1. Checks if JSON is valid and non-empty
    2. Extracts API calls from ALL processes
    3. Sorts them by timestamp (chronological order)
    4. Normalizes timestamps (relative to first call)
    5. Enriches with context (process, arguments)
    6. Optionally filters benign APIs
    7. Truncates if too long
    
    Args:
        json_ Parsed JSON dictionary from Cuckoo/CAPE
        max_calls: Maximum number of API calls to keep (truncate if longer)
        min_calls: Minimum number of API calls required (skip if shorter)
        filter_benign: Whether to filter out common benign APIs
        
    Returns:
        List of API call dictionaries in temporal order, or None if invalid
        
    Example output:
        [
            {
                "api": "NtCreateFile",
                "timestamp": 1234567890.123,
                "time_since_start": 0.0,
                "process_name": "malware.exe",
                "pid": 2844,
                "tid": 1234,
                "arguments": {...},
                "return_value": "0x00000000",
                "call_index": 0,
            },
            ...
        ]
    """
    
    # =========================================================================
    # 1. VALIDATE INPUT (Handle empty/malformed JSON)
    # =========================================================================
    
    # Check if data is None or empty
    if not json_:
        return None
    
    # Check if data is empty dict/list
    if isinstance(json_, dict) and len(json_) == 0:
        return None
    
    if isinstance(json_, list) and len(json_) == 0:
        return None
    
    # Check if behavior section exists
    if 'behavior' not in json_:
        return None
    
    behavior = json_['behavior']
    
    if not behavior or 'processes' not in behavior:
        return None
    
    processes = behavior['processes']
    
    if not processes or len(processes) == 0:
        return None
    
    # =========================================================================
    # 2. EXTRACT API CALLS FROM ALL PROCESSES
    # =========================================================================
    
    all_calls = []
    
    for process_idx, process in enumerate(processes):
        # Extract process metadata
        process_name = process.get('process_name', f'unknown_process_{process_idx}')
        process_id = process.get('process_id', process.get('pid', -1))
        parent_id = process.get('parent_id', process.get('ppid', -1))
        
        # Get API calls for this process
        if 'calls' not in process:
            continue
        
        calls = process['calls']
        
        if not calls or len(calls) == 0:
            continue
        
        # Extract each API call
        for call_idx, call in enumerate(calls):
            if not isinstance(call, dict):
                continue
            
            # Extract API name
            api_name = call.get('api', call.get('category', 'unknown'))
            
            # Skip if no API name
            if not api_name or api_name == 'unknown':
                continue
            
            # Filter benign APIs if enabled
            if filter_benign and api_name in BENIGN_APIS:
                continue
            
            # Extract timestamp (handle both Unix float and datetime string)
            timestamp = call.get('timestamp', call.get('time', 0.0))
            timestamp = parse_timestamp(timestamp)

            
            # Extract arguments
            arguments = call.get('arguments', [])
            
            # Extract return value
            return_value = call.get('return', call.get('return_value', None))
            
            # Extract thread ID
            thread_id = call.get('tid', call.get('thread_id', -1))
            
            # Create enriched call record
            call_record = {
                'api': api_name,
                'timestamp': float(timestamp),
                'process_name': process_name,
                'pid': process_id,
                'ppid': parent_id,
                'tid': thread_id,
                'arguments': arguments,
                'return_value': return_value,
                'process_idx': process_idx,
                'call_idx_in_process': call_idx,
            }
            
            all_calls.append(call_record)
    
    # =========================================================================
    # 3. CHECK MINIMUM CALLS REQUIREMENT
    # =========================================================================
    
    if len(all_calls) < min_calls:
        # Too few API calls, likely incomplete/corrupted sample
        return None
    
    # =========================================================================
    # 4. SORT BY TIMESTAMP (Create unified temporal sequence)
    # =========================================================================
    
    all_calls.sort(key=lambda x: x['timestamp'])
    
    # =========================================================================
    # 5. NORMALIZE TIMESTAMPS (Relative to execution start)
    # =========================================================================
    
    if len(all_calls) == 0:
        return None
    
    # First call timestamp = time 0
    start_time = all_calls[0]['timestamp']
    
    for idx, call in enumerate(all_calls):
        call['time_since_start'] = call['timestamp'] - start_time
        call['call_index'] = idx  # Global index in sequence
    
    # =========================================================================
    # 6. TRUNCATE IF TOO LONG (Memory management)
    # =========================================================================
    
    if len(all_calls) > max_calls:
        all_calls = all_calls[:max_calls]
    
    return all_calls


def extract_api_arguments(
    arguments: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Extract and structure API call arguments.
    
    Cuckoo arguments are often a list of dicts like:
    [
        {"name": "FileHandle", "value": "0x00000123"},
        {"name": "DesiredAccess", "value": "GENERIC_READ"},
        ...
    ]
    
    This converts to a clean dict:
    {
        "FileHandle": "0x00000123",
        "DesiredAccess": "GENERIC_READ",
        ...
    }
    
    Args:
        arguments: List of argument dictionaries from Cuckoo
        
    Returns:
        Dictionary mapping argument names to values
    """
    
    if not arguments:
        return {}
    
    arg_dict = {}
    
    for arg in arguments:
        if not isinstance(arg, dict):
            continue
        
        name = arg.get('name', arg.get('argument_name', None))
        value = arg.get('value', arg.get('argument_value', None))
        
        if name:
            arg_dict[name] = value
    
    return arg_dict


def get_api_category(api_name: str) -> str:
    """
    Categorize API call by function.
    
    Categories:
    - file: File system operations
    - registry: Registry operations
    - process: Process/thread operations
    - network: Network operations
    - crypto: Cryptography operations
    - memory: Memory operations
    - system: System information queries
    - other: Uncategorized
    
    Args:
        api_name: API function name
        
    Returns:
        Category string
    """
    
    api_lower = api_name.lower()
    
    # File operations
    if any(x in api_lower for x in ['file', 'directory', 'folder', 'read', 'write']):
        return 'file'
    
    # Registry operations
    if any(x in api_lower for x in ['reg', 'registry', 'key']):
        return 'registry'
    
    # Process/thread operations
    if any(x in api_lower for x in ['process', 'thread', 'create', 'terminate']):
        return 'process'
    
    # Network operations
    if any(x in api_lower for x in ['socket', 'connect', 'send', 'recv', 'http', 'internet']):
        return 'network'
    
    # Cryptography operations
    if any(x in api_lower for x in ['crypt', 'encrypt', 'decrypt', 'hash', 'cipher']):
        return 'crypto'
    
    # Memory operations
    if any(x in api_lower for x in ['memory', 'virtual', 'alloc', 'free', 'protect']):
        return 'memory'
    
    # System information
    if any(x in api_lower for x in ['query', 'information', 'system']):
        return 'system'
    
    return 'other'


def summarize_api_sequence(api_sequence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate statistics about an API sequence.
    
    Useful for debugging and dataset analysis.
    
    Args:
        api_sequence: Parsed API sequence
        
    Returns:
        Dictionary with statistics
    """
    
    if not api_sequence:
        return {
            'num_calls': 0,
            'duration': 0.0,
            'num_processes': 0,
            'num_unique_apis': 0,
        }
    
    # Count unique APIs
    unique_apis = set(call['api'] for call in api_sequence)
    
    # Count unique processes
    unique_processes = set(call['pid'] for call in api_sequence)
    
    # Calculate duration
    duration = api_sequence[-1]['time_since_start'] if api_sequence else 0.0
    
    # Count by category
    categories = {}
    for call in api_sequence:
        category = get_api_category(call['api'])
        categories[category] = categories.get(category, 0) + 1
    
    # Most common APIs
    api_counts = {}
    for call in api_sequence:
        api = call['api']
        api_counts[api] = api_counts.get(api, 0) + 1
    
    top_apis = sorted(api_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        'num_calls': len(api_sequence),
        'duration': duration,
        'num_processes': len(unique_processes),
        'num_unique_apis': len(unique_apis),
        'categories': categories,
        'top_apis': top_apis,
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_and_parse_json_file(
    json_file: Path,
    **kwargs
) -> Optional[List[Dict[str, Any]]]:
    """
    Load a JSON file and parse it.
    
    Convenience function that combines file loading + parsing.
    
    Args:
        json_file: Path to JSON file
        **kwargs: Additional arguments to parse_cape_sample
        
    Returns:
        Parsed API sequence or None
    """
    
    json_file = Path(json_file)
    
    if not json_file.exists():
        print(f"⚠️  File not found: {json_file}")
        return None
    
    # Check file size (skip empty files)
    if json_file.stat().st_size == 0:
        return None
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        return parse_cape_sample(data, **kwargs)
        
    except json.JSONDecodeError as e:
        print(f"⚠️  JSON decode error in {json_file.name}: {e}")
        return None
    
    except Exception as e:
        print(f"⚠️  Error parsing {json_file.name}: {e}")
        return None




# =============================================================================
# TESTING / DEBUGGING
# =============================================================================

def test_parse_on_sample(sample_file: str):
    """
    Test parsing on a single sample and print statistics.
    
    Usage:
        python -m data_processing.parse_cape_json
    """
    
    print("=" * 70)
    print("TESTING CAPE JSON PARSER")
    print("=" * 70)
    
    sample_path = Path(sample_file)
    
    if not sample_path.exists():
        print(f"❌ Sample file not found: {sample_file}")
        return False
    
    print(f"\n📄 Parsing: {sample_path.name}")
    
    # Parse
    api_sequence = load_and_parse_json_file(sample_path)
    
    if api_sequence is None:
        print("⚠️  Failed to parse (returned None)")
        print("   Reason: File is empty, malformed, or has no valid API calls")
        return False
    
    # Print statistics
    stats = summarize_api_sequence(api_sequence)
    
    print(f"\n✅ Successfully parsed!")
    print(f"\n📊 STATISTICS:")
    print(f"  Total API calls: {stats['num_calls']}")
    print(f"  Duration: {stats['duration']:.2f} seconds")
    print(f"  Unique processes: {stats['num_processes']}")
    print(f"  Unique APIs: {stats['num_unique_apis']}")
    
    print(f"\n📂 CATEGORY BREAKDOWN:")
    for category, count in sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {category:12} : {count:5} calls")
    
    print(f"\n🔝 TOP 10 APIS:")
    for api, count in stats['top_apis']:
        print(f"  {api:40} : {count:5} calls")
    
    print(f"\n📋 FIRST 5 CALLS:")
    for i, call in enumerate(api_sequence[:5]):
        print(f"\n  Call {i}:")
        print(f"    API: {call['api']}")
        print(f"    Time: {call['time_since_start']:.3f}s")
        print(f"    Process: {call['process_name']} (PID: {call['pid']})")
    
    print("\n" + "=" * 70)
    return True


if __name__ == "__main__":
    """
    Test the parser on sample files.
    
    This will try multiple files until it finds a valid one.
    
    Usage:
        # Test on first valid file found
        python -m data_processing.parse_cape_json
        
        # Test on specific file
        python -m data_processing.parse_cape_json path/to/file.json
    """
    
    import sys
    from config import RAW_SAMPLE_DIR
    
    # User provided a specific file
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        success = test_parse_on_sample(test_file)
        sys.exit(0 if success else 1)
    
    # Auto-find valid sample files
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
        
        success = test_parse_on_sample(str(json_file))
        
        if success:
            print("\n" + "=" * 70)
            print("✅ TESTING COMPLETE - Found valid sample!")
            print("=" * 70)
            print("\n💡 To test a specific file, run:")
            print(f"   python -m data_processing.parse_cape_json path/to/file.json")
            sys.exit(0)
        else:
            print(f"   → Skipping this file, trying next one...\n")
    
    # If we get here, no valid files found
    print("\n" + "=" * 70)
    print("⚠️  NO VALID FILES FOUND")
    print("=" * 70)
    print(f"\nTried {max_attempts} files, all were empty or malformed.")
    print("\nPossible issues:")
    print("  1. JSON files are empty")
    print("  2. JSON files are corrupted")
    print("  3. JSON files don't have 'behavior.processes.calls' structure")
    print("\nCheck your data/sample/ directory!")
    sys.exit(1)
