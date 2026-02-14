"""
Create Stratified Train/Val/Test Splits

This script creates stratified splits using extracted_metadata.csv as the source.

Features:
- Uses metadata.csv for faster processing
- Stratified splitting (maintains category ratios)
- Idempotent (checks if splits already exist)
- Reproducible (uses random seed)
- Detailed statistics output

Usage:
    # Standalone
    python -m data_processing.create_splits
    
    # From other scripts
    from data_processing.create_splits import create_stratified_splits
    create_stratified_splits()
"""

import random
from pathlib import Path
from collections import defaultdict, Counter
import json
import pandas as pd

# Import from config
from config import (
    RAW_SAMPLE_DIR,
    SPLITS_DIR,
    METADATA_FILE,
    CATEGORIES,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
)


def check_splits_exist(splits_dir=SPLITS_DIR):
    """
    Check if train/val/test split files already exist.
    
    Args:
        splits_dir: Directory containing split files
        
    Returns:
        bool: True if all split files exist, False otherwise
    """
    splits_dir = Path(splits_dir)
    
    required_files = [
        splits_dir / "train_samples.txt",
        splits_dir / "val_samples.txt",
        splits_dir / "test_samples.txt",
    ]
    
    return all(f.exists() for f in required_files)


def load_split_samples(split_name, splits_dir=SPLITS_DIR):
    """
    Load sample IDs from a split file.
    
    Args:
        split_name: "train", "val", or "test"
        splits_dir: Directory containing split files
        
    Returns:
        set: Set of sample IDs (filenames without extension)
    """
    splits_dir = Path(splits_dir)
    split_file = splits_dir / f"{split_name}_samples.txt"
    
    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}\n"
            f"Run create_stratified_splits() first!"
        )
    
    with open(split_file, 'r') as f:
        samples = {line.strip() for line in f if line.strip()}
    
    return samples


def load_metadata(metadata_file=METADATA_FILE):
    """
    Load and validate metadata CSV.
    
    Args:
        metadata_file: Path to extracted_metadata.csv
        
    Returns:
        pd.DataFrame: Loaded and validated metadata
    """
    metadata_path = Path(metadata_file)
    
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}\n"
            f"Please run the dataset organization script first to create:\n"
            f"  - data/sample/ (organized by category)\n"
            f"  - data/extracted_metadata.csv"
        )
    
    print(f"📄 Loading metadata from: {metadata_path}")
    df = pd.read_csv(metadata_path)
    
    # Validate required columns
    required_cols = ['filename', 'category_name']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(
            f"Metadata CSV missing required columns: {missing_cols}\n"
            f"Found columns: {list(df.columns)}"
        )
    
    # Remove .json extension if present
    df['sample_id'] = df['filename'].str.replace('.json', '', regex=False)
    
    print(f"✓ Loaded {len(df)} samples from metadata")
    
    return df


def create_stratified_splits(
    metadata_file=METADATA_FILE,
    output_dir=SPLITS_DIR,
    train_ratio=TRAIN_RATIO,
    val_ratio=VAL_RATIO,
    test_ratio=TEST_RATIO,
    random_seed=RANDOM_SEED,
    force_recreate=False,
    min_samples_per_category=10,
):
    """
    Create stratified train/val/test splits from metadata CSV.
    
    Ensures each split maintains the same category distribution as the full dataset.
    
    Args:
        metadata_file: Path to extracted_metadata.csv
        output_dir: Where to save split files (e.g., data/splits/)
        train_ratio: Fraction for training (default: 0.7)
        val_ratio: Fraction for validation (default: 0.15)
        test_ratio: Fraction for testing (default: 0.15)
        random_seed: Random seed for reproducibility
        force_recreate: If True, recreate splits even if they exist
        min_samples_per_category: Minimum samples required per category
        
    Returns:
        dict: Statistics about the created splits
    """
    
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    output_path = Path(output_dir)
    
    # Check if splits already exist
    if not force_recreate and check_splits_exist(output_path):
        print("=" * 70)
        print("SPLITS ALREADY EXIST")
        print("=" * 70)
        print(f"\n✓ Found existing splits in: {output_path}")
        
        # Load and print statistics
        train_samples = load_split_samples('train', output_path)
        val_samples = load_split_samples('val', output_path)
        test_samples = load_split_samples('test', output_path)
        
        print(f"\n📊 EXISTING SPLIT SIZES:")
        print(f"  Train: {len(train_samples)} samples")
        print(f"  Val:   {len(val_samples)} samples")
        print(f"  Test:  {len(test_samples)} samples")
        print(f"  Total: {len(train_samples) + len(val_samples) + len(test_samples)}")
        
        print("\n💡 To recreate splits, run with force_recreate=True")
        print("=" * 70)
        
        return {
            'train': len(train_samples),
            'val': len(val_samples),
            'test': len(test_samples),
        }
    
    # Set random seed
    random.seed(random_seed)
    
    print("=" * 70)
    print("CREATING STRATIFIED TRAIN/VAL/TEST SPLITS")
    print("=" * 70)
    print(f"\n📁 Metadata file: {metadata_file}")
    print(f"📁 Output directory: {output_path}")
    print(f"\n⚙️  Split ratios:")
    print(f"  Train: {train_ratio:.0%}")
    print(f"  Val:   {val_ratio:.0%}")
    print(f"  Test:  {test_ratio:.0%}")
    print(f"  Random seed: {random_seed}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load metadata
    df = load_metadata(metadata_file)
    
    # Group samples by category
    print(f"\n🔍 Grouping samples by category...")
    
    category_samples = {}
    category_counts = Counter()
    skipped_categories = []
    
    for category, group in df.groupby('category_name'):
        sample_ids = group['sample_id'].tolist()
        
        if len(sample_ids) < min_samples_per_category:
            print(f"  ⚠️  {category:15} : Only {len(sample_ids)} samples (skipping, need {min_samples_per_category}+)")
            skipped_categories.append(category)
            continue
        
        category_samples[category] = sample_ids
        category_counts[category] = len(sample_ids)
        
        print(f"  ✓ {category:15} : {len(sample_ids)} samples")
    
    if not category_samples:
        raise ValueError("No valid samples found in metadata!")
    
    total_samples = sum(category_counts.values())
    
    print(f"\n📊 DATASET SUMMARY:")
    print(f"  Valid categories: {len(category_samples)}")
    print(f"  Total samples: {total_samples}")
    if skipped_categories:
        print(f"  Skipped categories: {', '.join(skipped_categories)}")
    
    # Create stratified splits
    train_samples = []
    val_samples = []
    test_samples = []
    
    category_stats = {}
    
    print(f"\n📊 PER-CATEGORY SPLITS:")
    
    for category in sorted(category_samples.keys()):
        samples = category_samples[category]
        n_total = len(samples)
        
        # Calculate split sizes
        n_train = max(1, int(n_total * train_ratio))
        n_val = max(1, int(n_total * val_ratio))
        n_test = max(1, n_total - n_train - n_val)  # Ensure we use all samples
        
        # Shuffle samples for this category
        shuffled = samples.copy()
        random.shuffle(shuffled)
        
        # Split
        cat_train = shuffled[:n_train]
        cat_val = shuffled[n_train:n_train + n_val]
        cat_test = shuffled[n_train + n_val:]
        
        # Add to global splits
        train_samples.extend(cat_train)
        val_samples.extend(cat_val)
        test_samples.extend(cat_test)
        
        # Store stats
        category_stats[category] = {
            'total': n_total,
            'train': len(cat_train),
            'val': len(cat_val),
            'test': len(cat_test),
        }
        
        # Print
        print(f"\n  {category}:")
        print(f"    Total: {n_total}")
        print(f"    Train: {len(cat_train):4} ({len(cat_train)/n_total*100:5.1f}%)")
        print(f"    Val:   {len(cat_val):4} ({len(cat_val)/n_total*100:5.1f}%)")
        print(f"    Test:  {len(cat_test):4} ({len(cat_test)/n_total*100:5.1f}%)")
    
    # Shuffle the global splits (mix categories)
    random.shuffle(train_samples)
    random.shuffle(val_samples)
    random.shuffle(test_samples)
    
    # Save splits to files
    print(f"\n💾 Saving splits to {output_path}...")
    
    with open(output_path / 'train_samples.txt', 'w') as f:
        for sample_id in train_samples:
            f.write(f"{sample_id}\n")
    print(f"  ✓ train_samples.txt ({len(train_samples)} samples)")
    
    with open(output_path / 'val_samples.txt', 'w') as f:
        for sample_id in val_samples:
            f.write(f"{sample_id}\n")
    print(f"  ✓ val_samples.txt ({len(val_samples)} samples)")
    
    with open(output_path / 'test_samples.txt', 'w') as f:
        for sample_id in test_samples:
            f.write(f"{sample_id}\n")
    print(f"  ✓ test_samples.txt ({len(test_samples)} samples)")
    
    # Save detailed statistics as JSON
    stats = {
        'total_samples': total_samples,
        'train_samples': len(train_samples),
        'val_samples': len(val_samples),
        'test_samples': len(test_samples),
        'train_ratio': len(train_samples) / total_samples,
        'val_ratio': len(val_samples) / total_samples,
        'test_ratio': len(test_samples) / total_samples,
        'num_categories': len(category_samples),
        'categories': list(category_samples.keys()),
        'category_stats': category_stats,
        'random_seed': random_seed,
        'metadata_file': str(metadata_file),
    }
    
    with open(output_path / 'split_statistics.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"  ✓ split_statistics.json (detailed stats)")
    
    # Print final summary
    print("\n" + "=" * 70)
    print("SPLITS CREATED SUCCESSFULLY!")
    print("=" * 70)
    
    print(f"\n📊 FINAL SPLIT SIZES:")
    print(f"  Train: {len(train_samples):5} samples ({len(train_samples)/total_samples*100:.1f}%)")
    print(f"  Val:   {len(val_samples):5} samples ({len(val_samples)/total_samples*100:.1f}%)")
    print(f"  Test:  {len(test_samples):5} samples ({len(test_samples)/total_samples*100:.1f}%)")
    print(f"  ────────────────────────")
    print(f"  Total: {total_samples:5} samples")
    
    print(f"\n✅ Splits saved to: {output_path}")
    print("=" * 70)
    
    return stats


def verify_splits(splits_dir=SPLITS_DIR, metadata_file=METADATA_FILE):
    """
    Verify that split files are valid and samples exist in metadata.
    
    Args:
        splits_dir: Directory containing split files
        metadata_file: Path to extracted_metadata.csv
        
    Returns:
        dict: Verification results
    """
    print("=" * 70)
    print("VERIFYING SPLITS")
    print("=" * 70)
    
    splits_dir = Path(splits_dir)
    
    # Load metadata
    df = load_metadata(metadata_file)
    valid_sample_ids = set(df['sample_id'].values)
    
    print(f"✓ Metadata contains {len(valid_sample_ids)} valid samples")
    
    results = {
        'train': {'exists': 0, 'missing': 0, 'missing_ids': []},
        'val': {'exists': 0, 'missing': 0, 'missing_ids': []},
        'test': {'exists': 0, 'missing': 0, 'missing_ids': []},
    }
    
    for split_name in ['train', 'val', 'test']:
        print(f"\n🔍 Verifying {split_name} split...")
        
        samples = load_split_samples(split_name, splits_dir)
        
        for sample_id in samples:
            if sample_id in valid_sample_ids:
                results[split_name]['exists'] += 1
            else:
                results[split_name]['missing'] += 1
                results[split_name]['missing_ids'].append(sample_id)
        
        total = results[split_name]['exists'] + results[split_name]['missing']
        print(f"  ✓ Found: {results[split_name]['exists']}/{total}")
        
        if results[split_name]['missing'] > 0:
            print(f"  ⚠️  Missing: {results[split_name]['missing']}/{total}")
            print(f"     First 5 missing: {results[split_name]['missing_ids'][:5]}")
    
    # Check for overlap between splits
    print(f"\n🔍 Checking for overlap between splits...")
    train = load_split_samples('train', splits_dir)
    val = load_split_samples('val', splits_dir)
    test = load_split_samples('test', splits_dir)
    
    train_val_overlap = train & val
    train_test_overlap = train & test
    val_test_overlap = val & test
    
    if train_val_overlap:
        print(f"  ⚠️  Train-Val overlap: {len(train_val_overlap)} samples")
    else:
        print(f"  ✓ No Train-Val overlap")
    
    if train_test_overlap:
        print(f"  ⚠️  Train-Test overlap: {len(train_test_overlap)} samples")
    else:
        print(f"  ✓ No Train-Test overlap")
    
    if val_test_overlap:
        print(f"  ⚠️  Val-Test overlap: {len(val_test_overlap)} samples")
    else:
        print(f"  ✓ No Val-Test overlap")
    
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    
    return results


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Run this script to create splits:
        python -m data_processing.create_splits
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create stratified train/val/test splits from metadata"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreate splits even if they exist"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing splits"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help=f"Random seed (default: {RANDOM_SEED})"
    )
    
    args = parser.parse_args()
    
    if args.verify:
        # Verify existing splits
        verify_splits()
    else:
        # Create splits
        stats = create_stratified_splits(
            force_recreate=args.force,
            random_seed=args.seed,
        )
        
        print("\n✅ Done! You can now proceed to data loading and training.")
        print("\n💡 Next steps:")
        print("   1. Run verification: python -m data_processing.create_splits --verify")
        print("   2. Proceed to parse_cape_json.py implementation")
