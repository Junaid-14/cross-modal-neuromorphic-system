"""
Production training utilities for monitoring and optimization.

Provides:
- Cache integrity verification
- Memory usage estimation
- Optimal batch size detection
- Data loading profiling
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from tqdm import tqdm


def verify_cache_integrity(cache_path: Path) -> bool:
    """
    Verify that a cache file exists and is valid.

    Args:
        cache_path: Path to the cache file

    Returns:
        True if cache is valid, False otherwise
    """
    if not cache_path.exists():
        return False

    try:
        # Try to load the cache
        data = torch.load(cache_path)
        # Check if it's a tuple (train_data, test_data)
        if isinstance(data, tuple) and len(data) >= 2:
            train_data, test_data = data[:2]
            # Verify data is not empty
            if len(train_data) == 0 or len(test_data) == 0:
                return False
            return True
        return False
    except Exception as e:
        print(f"Cache verification failed: {e}")
        return False


def estimate_memory_usage(dataset_name: str, batch_size: int = 32) -> float:
    """
    Estimate RAM requirements for a dataset.

    Args:
        dataset_name: Name of dataset ('nmnist', 'shd', 'dvs_gesture', 'ssc')
        batch_size: Batch size for training

    Returns:
        Estimated memory in MB
    """
    # Approximate sizes based on dataset characteristics
    sizes = {
        'nmnist': {
            'train_samples': 60000,
            'test_samples': 10000,
            'train_samples_cached': 0,  # Streaming, not cached
            'test_samples_cached': 0,
            'sample_size_mb': 0.15,  # ~25 timesteps * 2 channels * 34*34
        },
        'shd': {
            'train_samples': 8156,
            'test_samples': 2268,
            'train_samples_cached': 8156,
            'test_samples_cached': 2268,
            'sample_size_mb': 0.28,  # ~100 timesteps * 700 channels
        },
        'dvs_gesture': {
            'train_samples': 12240,
            'test_samples': 2800,
            'train_samples_cached': 0,  # Streaming
            'test_samples_cached': 0,
            'sample_size_mb': 0.40,  # ~25 timesteps * 2 channels * 128*128
        },
        'ssc': {
            'train_samples': 30423,
            'test_samples': 4027,
            'train_samples_cached': 30423,
            'test_samples_cached': 4027,
            'sample_size_mb': 0.28,  # ~100 timesteps * 700 channels
        }
    }

    if dataset_name not in sizes:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    info = sizes[dataset_name]

    # Calculate cached data memory (if applicable)
    cached_memory = 0
    if info['train_samples_cached'] > 0:
        cached_memory = (
            info['train_samples_cached'] * info['sample_size_mb'] +
            info['test_samples_cached'] * info['sample_size_mb']
        )

    # Calculate batch memory
    batch_memory = batch_size * info['sample_size_mb']

    # Total memory estimate (cached + batch overhead + PyTorch overhead)
    total_memory = cached_memory + batch_memory * 2 + 100  # 100MB buffer

    return total_memory


def get_optimal_batch_size(
    dataset_name: str,
    model_variant: str = 'baseline',
    input_size: Tuple[int, int] = (32, 32),
    start_batch_size: int = 32,
    max_batch_size: int = 256,
    device: torch.device | None = None
) -> int:
    """
    Find the maximum batch size that fits in GPU memory.

    Args:
        dataset_name: Name of dataset
        model_variant: Model variant to test
        input_size: Input spatial dimensions
        start_batch_size: Starting batch size to test
        max_batch_size: Maximum batch size to try
        device: Device to test on (default: CUDA if available)

    Returns:
        Optimal batch size
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create a simple test model similar to actual models
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(2, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.fc1 = nn.Linear(128 * input_size[0] * input_size[1], 512)
            self.fc_out = nn.Linear(512, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            return self.fc_out(x)

    model = TestModel().to(device)

    for batch_size in range(start_batch_size, max_batch_size + 1, 32):
        try:
            # Create test input
            x = torch.randn(batch_size, 2, *input_size, device=device)

            # Forward pass
            output = model(x)

            # Try backward pass
            loss = output.sum()
            loss.backward()

            # Clean up
            del x, output, loss
            torch.cuda.empty_cache()

            print(f"✓ Batch size {batch_size} fits")

        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"✗ Batch size {batch_size} OOM, optimal is {batch_size - 32}")
                return batch_size - 32
            else:
                raise

    return max_batch_size


def monitor_data_loading(
    dataloader,
    num_batches: int = 10,
    description: str = "Data Loading"
) -> dict:
    """
    Profile data loading performance.

    Args:
        dataloader: DataLoader to monitor
        num_batches: Number of batches to sample
        description: Description for progress bar

    Returns:
        Dictionary with timing statistics
    """
    timings = []

    print(f"\nMonitoring {description} ({num_batches} batches)...")

    for batch_idx, (data, labels) in enumerate(tqdm(dataloader, total=num_batches, desc=description)):
        start = time.time()
        # Simulate processing (just moving to device)
        if torch.cuda.is_available():
            data = data.cuda()
            labels = labels.cuda()

        elapsed = time.time() - start
        timings.append(elapsed)

        if batch_idx >= num_batches - 1:
            break

    # Calculate statistics
    timings_array = torch.tensor(timings)
    results = {
        'mean_time_ms': timings_array.mean().item() * 1000,
        'std_time_ms': timings_array.std().item() * 1000,
        'min_time_ms': timings_array.min().item() * 1000,
        'max_time_ms': timings_array.max().item() * 1000,
        'median_time_ms': timings_array.median().item() * 1000,
        'batches_per_second': 1.0 / timings_array.mean().item(),
    }

    print(f"\n{description} Performance:")
    print(f"  Mean: {results['mean_time_ms']:.2f} ms")
    print(f"  Std:  {results['std_time_ms']:.2f} ms")
    print(f"  Min:  {results['min_time_ms']:.2f} ms")
    print(f"  Max:  {results['max_time_ms']:.2f} ms")
    print(f"  Throughput: {results['batches_per_second']:.1f} batches/sec")

    return results


def get_system_info() -> dict:
    """
    Get system information for debugging/optimization.

    Returns:
        Dictionary with system specs
    """
    info = {
        'cpu_count': os.cpu_count(),
        'cuda_available': torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info['cuda_total_memory_gb'] = props.total_memory / 1024**3
        info['cuda_compute_capability'] = f"{props.major}.{props.minor}"
    else:
        info['cuda_device_count'] = 0

    return info


def print_system_info() -> None:
    """Print formatted system information."""
    info = get_system_info()

    print("\n" + "="*60)
    print("SYSTEM INFORMATION")
    print("="*60)
    print(f"CPU Cores:    {info['cpu_count']}")
    print(f"CUDA Available: {info['cuda_available']}")

    if info['cuda_available']:
        print(f"CUDA Devices:  {info['cuda_device_count']}")
        print(f"CUDA Device:   {info['cuda_device_name']}")
        print(f"CUDA Memory:  {info['cuda_total_memory_gb']:.1f} GB")
        print(f"Compute Cap:   {info['cuda_compute_capability']}")

    print("="*60 + "\n")


if __name__ == "__main__":
    # Quick test
    print_system_info()

    # Test memory estimation
    for dataset in ['nmnist', 'shd', 'dvs_gesture', 'ssc']:
        mem_mb = estimate_memory_usage(dataset, batch_size=32)
        print(f"{dataset:15s}: ~{mem_mb:.1f} MB (batch_size=32)")
