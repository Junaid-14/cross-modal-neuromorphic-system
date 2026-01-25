"""
Dataset health check and verification utilities.

Provides:
- Dataset file verification
- Data corruption detection
- Dataset statistics reporting
- Quick data preview
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import tonic
from tqdm import tqdm


def verify_dataset_files(dataset_name: str, save_to: str = "./data") -> Dict[str, bool]:
    """
    Verify that all dataset files are present and accessible.

    Args:
        dataset_name: Name of dataset ('nmnist', 'shd', 'dvs_gesture', 'ssc')
        save_to: Directory where dataset is stored

    Returns:
        Dictionary with verification status for each file
    """
    save_path = Path(save_to)

    # Dataset-specific file patterns
    file_patterns = {
        'nmnist': [
            'NMNIST/train.npz',
            'NMNIST/test.npz',
        ],
        'shd': [
            'shd_train.h5',
            'shd_test.h5',
        ],
        'dvs_gesture': [
            'DVSGesture/train',
            'DVSGesture/test',
        ],
        'ssc': [
            'ssc_train.h5',
            'ssc_test.h5',
        ]
    }

    if dataset_name not in file_patterns:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    results = {}

    for pattern in file_patterns[dataset_name]:
        file_path = save_path / pattern
        results[pattern] = file_path.exists()

    return results


def check_data_integrity(
    train_loader,
    test_loader,
    num_samples_to_check: int = 100
) -> Dict[str, Any]:
    """
    Check for corrupted samples in the dataset.

    Args:
        train_loader: Training DataLoader
        test_loader: Testing DataLoader
        num_samples_to_check: Number of samples to check

    Returns:
        Dictionary with integrity check results
    """
    results = {
        'train_corrupted': 0,
        'test_corrupted': 0,
        'train_checked': 0,
        'test_checked': 0,
    }

    def check_dataloader(dataloader, key_prefix: str):
        count = 0
        corrupted = 0

        try:
            for data, labels in tqdm(dataloader, desc=f"Checking {key_prefix}", leave=False):
                try:
                    # Check for NaN or Inf values
                    if torch.isnan(data).any():
                        corrupted += 1
                    elif torch.isinf(data).any():
                        corrupted += 1
                    # Check data shape consistency
                    elif data.dim() == 0:
                        corrupted += 1
                    count += 1

                    if count >= num_samples_to_check:
                        break

                except Exception as e:
                    corrupted += 1
                    print(f"  Error at sample {count}: {e}")
                    count += 1

                    if count >= num_samples_to_check:
                        break

        except Exception as e:
            print(f"DataLoader error: {e}")

        return count, corrupted

    train_checked, train_corrupted = check_dataloader(train_loader, 'train')
    test_checked, test_corrupted = check_dataloader(test_loader, 'test')

    results['train_checked'] = train_checked
    results['test_checked'] = test_checked
    results['train_corrupted'] = train_corrupted
    results['test_corrupted'] = test_corrupted

    return results


def report_dataset_statistics(
    dataset_name: str,
    save_to: str = "./data",
    num_samples: int = 1000
) -> Dict[str, Any]:
    """
    Report statistics about the dataset.

    Args:
        dataset_name: Name of dataset
        save_to: Directory where dataset is stored
        num_samples: Number of samples to analyze

    Returns:
        Dictionary with dataset statistics
    """
    # Load dataset based on name
    try:
        if dataset_name.lower() == 'nmnist':
            dataset = tonic.datasets.NMNIST(save_to=save_to, train=True)
        elif dataset_name.lower() == 'shd':
            dataset = tonic.datasets.SHD(save_to=save_to, train=True)
        elif dataset_name.lower() == 'dvs_gesture' or dataset_name.lower() == 'dvsgesture':
            if hasattr(tonic.datasets, "DVS128Gesture"):
                dataset = tonic.datasets.DVS128Gesture(save_to=save_to, train=True)
            else:
                dataset = tonic.datasets.DVSGesture(save_to=save_to, train=True)
        elif dataset_name.lower() == 'ssc':
            dataset = tonic.datasets.SSC(save_to=save_to, train=True)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return {}

    # Collect statistics
    event_counts = []
    duration_counts = []
    label_counts = {}
    spatial_counts = []

    print(f"\nAnalyzing {num_samples} samples from {dataset_name}...")

    for idx in tqdm(range(min(num_samples, len(dataset))), desc="Analyzing"):
        events, label = dataset[idx]

        event_counts.append(len(events))
        duration_counts.append(events['t'].max() if len(events) > 0 else 0)
        label_counts[str(label)] = label_counts.get(str(label), 0) + 1

        # Spatial statistics
        if 'x' in events.dtype.names and len(events) > 0:
            spatial_counts.append((events['x'].max() - events['x'].min() + 1))
        if 'y' in events.dtype.names and len(events) > 0:
            spatial_counts.append((events['y'].max() - events['y'].min() + 1))

    # Calculate statistics
    stats = {
        'dataset_name': dataset_name,
        'total_samples': len(dataset),
        'samples_analyzed': min(num_samples, len(dataset)),
        'event_count_mean': np.mean(event_counts),
        'event_count_std': np.std(event_counts),
        'event_count_min': np.min(event_counts),
        'event_count_max': np.max(event_counts),
        'duration_mean_ms': np.mean(duration_counts) / 1000,
        'duration_std_ms': np.std(duration_counts) / 1000,
        'label_distribution': label_counts,
        'num_classes': len(label_counts),
        'class_balance': {
            cls: count / sum(label_counts.values())
            for cls, count in label_counts.items()
        }
    }

    if spatial_counts:
        stats['spatial_extent_mean'] = np.mean(spatial_counts)
        stats['spatial_extent_std'] = np.std(spatial_counts)

    return stats


def print_dataset_stats(stats: Dict[str, Any]) -> None:
    """
    Print formatted dataset statistics.

    Args:
        stats: Statistics dictionary from report_dataset_statistics
    """
    print("\n" + "="*60)
    print(f"DATASET STATISTICS: {stats.get('dataset_name', 'Unknown').upper()}")
    print("="*60)

    print(f"\nDataset Size:")
    print(f"  Total samples:  {stats.get('total_samples', 'N/A')}")
    print(f"  Analyzed:       {stats.get('samples_analyzed', 'N/A')}")

    if 'event_count_mean' in stats:
        print(f"\nEvent Statistics:")
        print(f"  Mean events/sample:  {stats['event_count_mean']:.1f}")
        print(f"  Std:                {stats['event_count_std']:.1f}")
        print(f"  Min:                {stats['event_count_min']}")
        print(f"  Max:                {stats['event_count_max']}")

    if 'duration_mean_ms' in stats:
        print(f"\nDuration Statistics:")
        print(f"  Mean duration:  {stats['duration_mean_ms']:.1f} ms")
        print(f"  Std:             {stats['duration_std_ms']:.1f} ms")

    print(f"\nClass Distribution:")
    label_dist = stats.get('label_distribution', {})
    for label, count in sorted(label_dist.items(), key=lambda x: int(x[0])):
        print(f"  Class {label}: {count:5d} samples ({label_dist[str(label)]:.1%})")

    print(f"\nTotal Classes: {stats.get('num_classes', 'N/A')}")

    print("="*60)


def quick_preview(dataset_name: str, save_to: str = "./data", num_samples: int = 5) -> None:
    """
    Quick preview of dataset samples.

    Args:
        dataset_name: Name of dataset
        save_to: Directory where dataset is stored
        num_samples: Number of samples to preview
    """
    try:
        if dataset_name.lower() == 'nmnist':
            dataset = tonic.datasets.NMNIST(save_to=save_to, train=True)
        elif dataset_name.lower() == 'shd':
            dataset = tonic.datasets.SHD(save_to=save_to, train=True)
        elif dataset_name.lower() == 'dvs_gesture' or dataset_name.lower() == 'dvsgesture':
            if hasattr(tonic.datasets, "DVS128Gesture"):
                dataset = tonic.datasets.DVS128Gesture(save_to=save_to, train=True)
            else:
                dataset = tonic.datasets.DVSGesture(save_to=save_to, train=True)
        elif dataset_name.lower() == 'ssc':
            dataset = tonic.datasets.SSC(save_to=save_to, train=True)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print(f"\n{'='*60}")
    print(f"QUICK PREVIEW: {dataset_name.upper()}")
    print(f"{'='*60}")

    for idx in range(min(num_samples, len(dataset))):
        events, label = dataset[idx]
        print(f"\nSample {idx + 1} (Label: {label}):")
        print(f"  Event count: {len(events)}")

        if len(events) > 0:
            print(f"  Time range: {events['t'].min():.0f} - {events['t'].max():.0f} µs")

            # Show first few events
            print(f"  First 3 events:")
            for i, event in enumerate(events[:3]):
                event_dict = {name: event[name].item() for name in event.dtype.names}
                print(f"    {i+1}: {event_dict}")

    print(f"\n{'='*60}\n")


def full_health_check(
    dataset_name: str,
    save_to: str = "./data",
    train_loader = None,
    test_loader = None
) -> Dict[str, Any]:
    """
    Run a comprehensive health check on the dataset.

    Args:
        dataset_name: Name of dataset
        save_to: Directory where dataset is stored
        train_loader: Optional training DataLoader
        test_loader: Optional testing DataLoader

    Returns:
        Dictionary with complete health check results
    """
    results = {
        'dataset_name': dataset_name,
        'files_ok': True,
        'data_ok': True,
    }

    # Check files
    print("\n" + "="*60)
    print("STEP 1: Checking dataset files...")
    print("="*60)
    file_status = verify_dataset_files(dataset_name, save_to)

    all_files_present = all(file_status.values())
    results['files_ok'] = all_files_present

    for file, exists in file_status.items():
        status = "✓" if exists else "✗"
        print(f"{status} {file}")

    if not all_files_present:
        results['data_ok'] = False
        print(f"\n⚠️  Some files are missing. Dataset may not be complete.")
        return results

    # Check data integrity
    if train_loader is not None and test_loader is not None:
        print("\n" + "="*60)
        print("STEP 2: Checking data integrity...")
        print("="*60)
        integrity_results = check_data_integrity(train_loader, test_loader)

        results.update(integrity_results)

        train_rate = integrity_results['train_corrupted'] / integrity_results['train_checked'] if integrity_results['train_checked'] > 0 else 0
        test_rate = integrity_results['test_corrupted'] / integrity_results['test_checked'] if integrity_results['test_checked'] > 0 else 0

        print(f"\nTrain set: {integrity_results['train_checked']} checked, {integrity_results['train_corrupted']} corrupted ({train_rate:.1%})")
        print(f"Test set:  {integrity_results['test_checked']} checked, {integrity_results['test_corrupted']} corrupted ({test_rate:.1%})")

        if train_rate > 0.05 or test_rate > 0.05:
            results['data_ok'] = False
            print(f"\n⚠️  High corruption rate detected!")

    # Report statistics
    print("\n" + "="*60)
    print("STEP 3: Gathering statistics...")
    print("="*60)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stats = report_dataset_statistics(dataset_name, save_to)

    results['statistics'] = stats

    # Final verdict
    print("\n" + "="*60)
    print("HEALTH CHECK SUMMARY")
    print("="*60)

    if results['files_ok'] and results['data_ok']:
        print("✅ Dataset is healthy and ready for training!")
    elif not results['files_ok']:
        print("❌ Dataset files are missing. Please download the dataset.")
    elif not results['data_ok']:
        print("❌ Dataset has corruption issues. Check for damaged samples.")
    else:
        print("⚠️  Dataset has minor issues. Proceed with caution.")

    print("="*60 + "\n")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dataset health check utilities")
    parser.add_argument('--dataset', type=str, choices=['nmnist', 'shd', 'dvs_gesture', 'ssc'],
                        help='Dataset to check')
    parser.add_argument('--save-to', type=str, default='./data',
                        help='Dataset directory')
    parser.add_argument('--stats-only', action='store_true',
                        help='Only show statistics, skip health check')

    args = parser.parse_args()

    if args.stats_only:
        stats = report_dataset_statistics(args.dataset, args.save_to)
        print_dataset_stats(stats)
    else:
        full_health_check(args.dataset, args.save_to)
