"""
DVS-Gesture Dataset Loader
Event-based recordings of hand/arm gestures (DVS128 Gesture).
"""

from __future__ import annotations

import os
from typing import Callable, Tuple

import torch
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F


def _get_dvs_gesture_dataset_class() -> Callable:
    """
    Resolve the DVS-Gesture dataset class across tonic versions.
    """
    if hasattr(tonic.datasets, "DVS128Gesture"):
        return tonic.datasets.DVS128Gesture
    if hasattr(tonic.datasets, "DVSGesture"):
        return tonic.datasets.DVSGesture
    raise AttributeError("tonic.datasets missing DVS-Gesture dataset class")


def get_dvs_gesture_loaders(
    batch_size: int = 16,
    time_steps: int = 25,
    num_workers: int | None = None,
    pin_memory: bool | None = None,
    persistent_workers: bool = True,
    save_to: str = "./data",
    time_window: int = 1000,
    target_spatial_size: Tuple[int, int] = (128, 128),
) -> Tuple[DataLoader, DataLoader]:
    """
    Get DVS-Gesture train and test data loaders.

    Args:
        batch_size: Batch size for training
        time_steps: Number of time steps for temporal encoding
        num_workers: Number of worker processes for data loading
        save_to: Directory to download/store the dataset
        time_window: Window size (in microseconds) for event framing
        target_spatial_size: Optional spatial resize to match model input

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

    dataset_cls = _get_dvs_gesture_dataset_class()
    sensor_size = dataset_cls.sensor_size

    frame_transform = transforms.Compose(
        [
            transforms.Denoise(filter_time=10000),
            transforms.ToFrame(sensor_size=sensor_size, time_window=time_window),
        ]
    )

    train_dataset = dataset_cls(save_to=save_to, transform=frame_transform, train=True)
    test_dataset = dataset_cls(save_to=save_to, transform=frame_transform, train=False)

    def collate_fn(batch):
        """Optimized collation function with reduced tensor operations."""
        frames, labels = zip(*batch)

        # Helper function to normalize and resize frames
        def _process_frame(frame):
            tensor = torch.as_tensor(frame, dtype=torch.float32)
            # Handle different channel arrangements
            if tensor.dim() == 4 and tensor.shape[1] not in (1, 2):
                if tensor.shape[-1] in (1, 2):
                    tensor = tensor.permute(0, 3, 1, 2)
            # Resize if needed
            if target_spatial_size is not None:
                tensor = F.interpolate(tensor, size=target_spatial_size, mode="nearest")
            return tensor

        # Process all frames and pad/truncate to time_steps
        frames_list = []
        for f in frames:
            f_tensor = _process_frame(f)
            # Pad or truncate to time_steps
            if f_tensor.shape[0] >= time_steps:
                frames_list.append(f_tensor[:time_steps])
            else:
                padding = torch.zeros(
                    time_steps - f_tensor.shape[0], *f_tensor.shape[1:], dtype=torch.float32
                )
                frames_list.append(torch.cat([f_tensor, padding], dim=0))

        # Stack all frames at once (faster than zero-allocation + loop)
        frames_tensor = torch.stack(frames_list, dim=0)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return frames_tensor, labels_tensor

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

    print(
        f"✓ DVS-Gesture loaded: {len(train_dataset)} train, {len(test_dataset)} test"
    )
    return train_loader, test_loader
