"""
N-MNIST Dataset Loader
Event-camera recordings of handwritten digits
"""

import os
import torch
import tonic
from torch.utils.data import DataLoader
import tonic.transforms as transforms


def get_nmnist_loaders(
    batch_size=32,
    time_steps=25,
    num_workers=None,
    pin_memory=None,
    persistent_workers=True,
):
    """
    Get N-MNIST train and test data loaders

    Args:
        batch_size: Batch size for training
        time_steps: Number of time steps for temporal encoding
        num_workers: Number of worker processes (default: min(8, os.cpu_count()))
        pin_memory: If True, use pinned memory for faster GPU transfer
                   (default: True for GPU, False for CPU)
        persistent_workers: If True, workers stay alive between epochs (default: True)

    Returns:
        train_loader, test_loader: PyTorch DataLoader objects
    """
    # Auto-detect optimal settings
    if num_workers is None:
        num_workers = min(8, os.cpu_count())

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    # Disable persistent_workers for single worker (PyTorch limitation)
    if num_workers == 0 or num_workers == 1:
        persistent_workers = False
    
    # Define transforms
    sensor_size = tonic.datasets.NMNIST.sensor_size
    frame_transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(
            sensor_size=sensor_size,
            time_window=1000
        ),
    ])
    
    # Load datasets
    train_dataset = tonic.datasets.NMNIST(
        save_to='./data',
        transform=frame_transform,
        train=True
    )

    test_dataset = tonic.datasets.NMNIST(
        save_to='./data',
        transform=frame_transform,
        train=False
    )
    
    def collate_fn(batch):
        """Optimized collation function with reduced tensor operations."""
        frames, labels = zip(*batch)

        # Process all frames in a single loop
        frames_list = []
        for f in frames:
            f_tensor = torch.as_tensor(f, dtype=torch.float32)
            # Pad or truncate to time_steps
            if f_tensor.shape[0] >= time_steps:
                frames_list.append(f_tensor[:time_steps])
            else:
                # Pad with zeros
                padding = torch.zeros(
                    time_steps - f_tensor.shape[0], *f_tensor.shape[1:], dtype=torch.float32
                )
                frames_list.append(torch.cat([f_tensor, padding], dim=0))

        # Stack all frames at once (faster than zero-allocation + loop)
        frames_tensor = torch.stack(frames_list, dim=0)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return frames_tensor, labels_tensor

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    
    print(f"✓ N-MNIST loaded: {len(train_dataset)} train, {len(test_dataset)} test")
    return train_loader, test_loader
