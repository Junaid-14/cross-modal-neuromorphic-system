#!/usr/bin/env python3
"""
Standalone CLI to run the research sweep on CUDA, MPS, or CPU.

Sweep:
- Datasets: N-Caltech101 (vision), SHD (20-class audio)
- Configurations: baseline SNN vs structural plasticity (breeding/killing)
- Metrics: accuracy mean/std, temporal silhouette, SynOps/FLOPs energy ratio

Outputs:
- `experiment_log.txt` (terminal + file logging)
- JSON summaries in `results/`
- Temporal engram plots in `results/`
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

# Force non-interactive plotting so the script can run detached from an IDE.
import matplotlib

matplotlib.use("Agg")

from Dataloaders.shd_loader import get_shd_loaders
from Models.registry import build_model
from training.pipeline import (
    make_energy_methodology_callback,
    make_temporal_silhouette_callback,
    run_seed_iterator,
    train_model,
)
from training.settings import TrainingConfig
from visualization import plot_engram_evolution


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = ROOT / "results"
DEFAULT_LOG_FILE = ROOT / "experiment_log.txt"


def configure_logging(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("research_sweep")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return _json_safe(obj.item())
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        if not torch.isfinite(torch.tensor(obj)):
            return None
    return obj


def _format_mean_std(stats: Dict[str, Any], decimals: int = 3, suffix: str = "") -> str:
    if not isinstance(stats, dict):
        return "n/a"
    mean = stats.get("mean")
    std = stats.get("std_dev", stats.get("std"))
    if mean is None:
        return "n/a"
    if isinstance(mean, (int, float)) and isinstance(std, (int, float)):
        return f"{mean:.{decimals}f} +/- {std:.{decimals}f}{suffix}"
    return "n/a"


def _temporal_stat(final_specs: Dict[str, Any], key: str) -> Dict[str, Any]:
    temporal = final_specs.get("temporal_silhouette", {})
    if isinstance(temporal, dict) and key in temporal:
        return temporal.get(key, {})
    return {}


class _NCaltechCachedFramesDataset(Dataset):
    """Thin dataset wrapper so train/test subsets can share converted frame tensors."""

    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        return self.base_dataset[idx]


class _NCaltechLabelMapWrapper(Dataset):
    """Maps string class labels to integer indices for N-Caltech101."""

    def __init__(self, base_dataset: Dataset, label_to_idx: dict):
        self.base_dataset = base_dataset
        self.label_to_idx = label_to_idx

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int):
        frames, label = self.base_dataset[idx]
        return frames, self.label_to_idx[label]


def _to_tchw_frames(frames: Any) -> torch.Tensor:
    x = torch.as_tensor(frames, dtype=torch.float32)
    if x.dim() != 4:
        raise ValueError(f"Expected 4D frame tensor [T,C,H,W] or [T,H,W,C], got {tuple(x.shape)}")
    # [T, C, H, W]
    if x.size(1) in (1, 2):
        return x
    # [T, H, W, C]
    if x.size(-1) in (1, 2):
        return x.permute(0, 3, 1, 2).contiguous()
    raise ValueError(f"Could not infer channel dimension for frame tensor with shape {tuple(x.shape)}")


def _resize_and_pad_event_frames(
    frames: Any,
    time_steps: int,
    spatial_size: Tuple[int, int],
    expected_channels: int = 2,
) -> torch.Tensor:
    x = _to_tchw_frames(frames)

    if x.size(1) < expected_channels:
        pad_ch = expected_channels - x.size(1)
        x = torch.cat(
            [x, torch.zeros(x.size(0), pad_ch, x.size(2), x.size(3), dtype=x.dtype)],
            dim=1,
        )
    elif x.size(1) > expected_channels:
        x = x[:, :expected_channels]

    if tuple(x.shape[-2:]) != tuple(spatial_size):
        x = F.interpolate(x, size=spatial_size, mode="bilinear", align_corners=False)

    out = torch.zeros(time_steps, expected_channels, spatial_size[0], spatial_size[1], dtype=torch.float32)
    length = min(int(x.size(0)), int(time_steps))
    if length > 0:
        out[:length] = x[:length]
    return out


def _make_event_frame_collate_fn(time_steps: int, spatial_size: Tuple[int, int], expected_channels: int = 2):
    def collate_fn(batch):
        frames, labels = zip(*batch)
        padded = torch.stack(
            [
                _resize_and_pad_event_frames(
                    f, time_steps=time_steps, spatial_size=spatial_size, expected_channels=expected_channels
                )
                for f in frames
            ],
            dim=0,
        )
        labels_tensor = torch.as_tensor(labels, dtype=torch.long)
        return padded, labels_tensor

    return collate_fn


def _get_ncaltech_dataset_class(tonic_module):
    candidate_names = [
        "NCALTECH101",
        "NCaltech101",
        "NCALTECH_101",
        "N_CALTECH101",
    ]
    for name in candidate_names:
        dataset_cls = getattr(tonic_module.datasets, name, None)
        if dataset_cls is not None:
            return dataset_cls, name
    raise AttributeError(
        f"Could not find N-Caltech101 dataset class in tonic.datasets. Tried: {candidate_names}"
    )


def get_ncaltech101_loaders(
    batch_size: int,
    num_workers: int,
    data_dir: Path,
    time_steps: int,
    spatial_size: Tuple[int, int],
    train_split: float,
    split_seed: int,
    frame_time_window_us: int,
    logger: logging.Logger,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build deterministic train/test loaders for N-Caltech101 using a random split.
    """
    try:
        import tonic
        import tonic.transforms as tonic_transforms
    except ImportError as exc:
        raise RuntimeError(
            "Tonic is required for N-Caltech101. Install dependencies (e.g., `pip install tonic`)."
        ) from exc

    dataset_cls, dataset_cls_name = _get_ncaltech_dataset_class(tonic)
    logger.info("Using tonic.datasets.%s for N-Caltech101", dataset_cls_name)

    sensor_size = getattr(dataset_cls, "sensor_size", None)
    if sensor_size is None:
        # Fallback if dataset class exposes sensor_size only on instances.
        temp_ds = dataset_cls(save_to=str(data_dir))
        sensor_size = temp_ds.sensor_size

    frame_transform = tonic_transforms.Compose(
        [
            tonic_transforms.Denoise(filter_time=10000),
            tonic_transforms.ToFrame(sensor_size=sensor_size, time_window=frame_time_window_us),
        ]
    )

    # Build label map from a raw dataset (no transform) to avoid invoking event transforms
    # during the label scan. Some tonic versions can fail in Denoise(...) for malformed samples.
    raw_dataset = dataset_cls(save_to=str(data_dir))
    # N-Caltech101 returns string labels; map to integer indices.
    all_labels = set()
    for i in range(len(raw_dataset)):
        _, lb = raw_dataset[i]
        all_labels.add(lb)
    label_to_idx = {s: i for i, s in enumerate(sorted(all_labels))}

    # N-Caltech101 typically has no train/test split; create a deterministic one.
    full_dataset = dataset_cls(save_to=str(data_dir), transform=frame_transform)
    full_dataset = _NCaltechLabelMapWrapper(full_dataset, label_to_idx)
    full_dataset = _NCaltechCachedFramesDataset(full_dataset)

    total_len = len(full_dataset)
    train_len = int(total_len * train_split)
    test_len = total_len - train_len
    split_gen = torch.Generator().manual_seed(split_seed)
    train_dataset, test_dataset = random_split(full_dataset, [train_len, test_len], generator=split_gen)

    collate_fn = _make_event_frame_collate_fn(
        time_steps=time_steps,
        spatial_size=spatial_size,
        expected_channels=2,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        collate_fn=collate_fn,
    )

    logger.info(
        "N-Caltech101 ready: %d total | %d train | %d test | T=%d | spatial=%s",
        total_len,
        train_len,
        test_len,
        time_steps,
        spatial_size,
    )
    return train_loader, test_loader


def build_device(logger: logging.Logger, requested_device: str) -> torch.device:
    requested = requested_device.lower().strip()
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()

    if requested == "auto":
        if cuda_available:
            device = torch.device("cuda")
        elif mps_available:
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    elif requested == "cuda":
        if not cuda_available:
            raise RuntimeError("CUDA was requested (`--device cuda`) but no CUDA device is available.")
        device = torch.device("cuda")
    elif requested == "mps":
        if not mps_available:
            raise RuntimeError("MPS was requested (`--device mps`) but MPS is not available.")
        device = torch.device("mps")
    elif requested == "cpu":
        device = torch.device("cpu")
    else:
        raise ValueError(f"Unsupported device selection: {requested_device}")

    logger.info("Using device: %s", device)
    if device.type == "cuda":
        gpu_idx = 0 if device.index is None else device.index
        try:
            gpu_name = torch.cuda.get_device_name(gpu_idx)
            props = torch.cuda.get_device_properties(gpu_idx)
            logger.info(
                "CUDA GPU: %s | compute capability %d.%d | VRAM %.1f GB",
                gpu_name,
                props.major,
                props.minor,
                props.total_memory / (1024**3),
            )
        except Exception:
            logger.info("CUDA GPU detected, but device properties could not be queried.")

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    return device


def _build_markdown_summary_table(records: List[Dict[str, Any]]) -> str:
    headers = [
        "Config",
        "Dataset",
        "Accuracy (mean+/-std)",
        "t25 Sil",
        "t50 Sil",
        "t75 Sil",
        "t100 Sil",
        "SynOps/sample",
        "FLOPs/sample",
        "Energy Ratio",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for rec in records:
        final_specs = rec["final_specs"]
        compute = final_specs.get("compute_comparison", {})
        mean_ops = compute.get("mean_synops_vs_mean_flops_per_sample", {})
        row = [
            rec["config_name"],
            rec["dataset_name"],
            _format_mean_std(final_specs.get("accuracy", {}), decimals=2),
            _format_mean_std(_temporal_stat(final_specs, "t25"), decimals=3),
            _format_mean_std(_temporal_stat(final_specs, "t50"), decimals=3),
            _format_mean_std(_temporal_stat(final_specs, "t75"), decimals=3),
            _format_mean_std(_temporal_stat(final_specs, "t100"), decimals=3),
            _format_mean_std(mean_ops.get("snn_mean_synops_per_sample", {}), decimals=1),
            _format_mean_std(mean_ops.get("ann_flops_per_sample", {}), decimals=1),
            _format_mean_std(compute.get("energy_improvement_ratio", {}), decimals=2),
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def _build_training_config(
    args: argparse.Namespace,
    checkpoint_dir: Path,
    enable_structural_plasticity: bool,
) -> TrainingConfig:
    return TrainingConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        contrastive_weight=args.contrastive_weight,
        contrastive_temperature=args.contrastive_temperature,
        gradient_clip=args.gradient_clip,
        num_epochs=args.num_epochs,
        patience=args.patience,
        checkpoint_dir=checkpoint_dir,
        enable_structural_plasticity=enable_structural_plasticity,
        structural_plasticity_interval=args.sp_interval,
        structural_plasticity_max_batches=args.sp_profile_batches,
        structural_plasticity_kill_start_epoch=args.sp_kill_start_epoch,
        structural_plasticity_kill_threshold=args.sp_kill_threshold,
        structural_plasticity_alpha_frac=args.sp_alpha_frac,
        structural_plasticity_breed_sigma=args.sp_breed_sigma,
    )


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2)


def run_single_sweep(
    *,
    config_name: str,
    dataset_name: str,
    modality: str,
    input_type: str,
    num_classes: int,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    results_dir: Path,
    logger: logging.Logger,
    spatial_size: Tuple[int, int] | None = None,
    input_size: int | None = None,
    enable_structural_plasticity: bool,
) -> Dict[str, Any]:
    tag = f"{config_name}__{dataset_name}".replace(" ", "_").lower()
    checkpoint_dir = results_dir / "checkpoints" / tag
    config = _build_training_config(args, checkpoint_dir, enable_structural_plasticity)

    logger.info(
        "Starting sweep: config=%s | dataset=%s | modality=%s | num_classes=%d",
        config_name,
        dataset_name,
        modality,
        num_classes,
    )

    temporal_cb = make_temporal_silhouette_callback(
        val_loader=test_loader,
        device=device,
        max_samples=args.temporal_max_samples,
    )
    energy_cb = make_energy_methodology_callback(
        eval_loader=test_loader,
        device=device,
        profile_max_batches=(None if args.energy_profile_max_batches <= 0 else args.energy_profile_max_batches),
        ann_energy_per_flop_pj=args.ann_energy_per_flop_pj,
        snn_energy_per_synop_pj=args.snn_energy_per_synop_pj,
    )

    def train_once(seed: int):
        model = build_model(
            variant="baseline",
            input_type=input_type,
            num_classes=num_classes,
            spatial_size=spatial_size or (34, 34),
            input_size=input_size or 700,
        ).to(device)

        model_name = f"{tag}_seed{seed}"
        return train_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            model_name=model_name,
            dataset_name=dataset_name,
            use_contrastive=False,
            device=device,
            config=config,
        )

    run_result = run_seed_iterator(
        train_once_fn=train_once,
        num_runs=args.num_runs,
        base_seed=args.base_seed,
        silhouette_fn=temporal_cb,
        energy_fn=energy_cb,
        deterministic=True,
        final_specs_path=results_dir / f"{tag}__final_specs.json",
    )

    compact_payload = {
        "config_name": config_name,
        "dataset_name": dataset_name,
        "modality": modality,
        "input_type": input_type,
        "num_classes": num_classes,
        "num_runs": run_result["num_runs"],
        "seeds": run_result["seeds"],
        "summary": run_result["summary"],
        "temporal_silhouette_summary": run_result["temporal_silhouette_summary"],
        "energy_summary": run_result.get("energy_summary", {}),
        "final_specs": run_result.get("final_specs", {}),
        "final_specs_path": run_result.get("final_specs_path"),
    }
    _save_json(results_dir / f"{tag}__summary.json", compact_payload)

    logger.info(
        "Completed sweep: %s | Acc=%s | Energy Ratio=%s",
        tag,
        _format_mean_std(run_result.get("summary", {}).get("accuracy", {}), decimals=2),
        _format_mean_std(run_result.get("summary", {}).get("energy.ratio", {}), decimals=2),
    )
    return compact_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full SNN research sweep on CUDA, MPS, or CPU.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device to use. `auto` prefers CUDA, then MPS, then CPU.",
    )
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0, help="Use 0 on macOS for safer standalone runs.")

    # Training config
    parser.add_argument("--num-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--contrastive-temperature", type=float, default=0.07)
    parser.add_argument("--gradient-clip", type=float, default=1.0)

    # Structural plasticity (breeding/killing) settings
    parser.add_argument("--sp-interval", type=int, default=5)
    parser.add_argument("--sp-profile-batches", type=int, default=50)
    parser.add_argument("--sp-kill-start-epoch", type=int, default=10)
    parser.add_argument("--sp-kill-threshold", type=float, default=0.01)
    parser.add_argument("--sp-alpha-frac", type=float, default=0.05)
    parser.add_argument("--sp-breed-sigma", type=float, default=0.01)

    # Temporal evaluation
    parser.add_argument("--temporal-max-samples", type=int, default=512)

    # Energy evaluation
    parser.add_argument("--energy-profile-max-batches", type=int, default=0, help="<=0 means full eval loader.")
    parser.add_argument("--ann-energy-per-flop-pj", type=float, default=100.0)
    parser.add_argument("--snn-energy-per-synop-pj", type=float, default=0.1)

    # N-Caltech101 settings
    parser.add_argument("--ncaltech-time-steps", type=int, default=25)
    parser.add_argument("--ncaltech-frame-time-window-us", type=int, default=1000)
    parser.add_argument("--ncaltech-height", type=int, default=34)
    parser.add_argument("--ncaltech-width", type=int, default=34)
    parser.add_argument("--ncaltech-train-split", type=float, default=0.8)
    parser.add_argument("--ncaltech-split-seed", type=int, default=123)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["ncaltech101", "shd"],
        default=["shd"],
        help="Datasets to run (example: --datasets shd).",
    )

    parser.add_argument("--timestamped-output", action="store_true", help="Prefix aggregate filenames with timestamp.")
    return parser.parse_args()


def main() -> int:
    os.chdir(ROOT)
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)

    logger = configure_logging(DEFAULT_LOG_FILE)
    logger.info("Standalone research sweep started.")
    logger.info("Working directory: %s", ROOT)
    logger.info("Results directory: %s", args.results_dir)

    device = build_device(logger, args.device)

    # Prepare dataset loaders once per dataset; reused across seeds/configs.
    logger.info("Preparing dataset loaders...")
    vision_spatial = (args.ncaltech_height, args.ncaltech_width)
    selected_datasets = set(args.datasets)
    ncaltech_train_loader = ncaltech_test_loader = None
    shd_train_loader = shd_test_loader = None

    if "ncaltech101" in selected_datasets:
        ncaltech_train_loader, ncaltech_test_loader = get_ncaltech101_loaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            data_dir=args.data_dir,
            time_steps=args.ncaltech_time_steps,
            spatial_size=vision_spatial,
            train_split=args.ncaltech_train_split,
            split_seed=args.ncaltech_split_seed,
            frame_time_window_us=args.ncaltech_frame_time_window_us,
            logger=logger,
        )
    else:
        logger.info("Skipping N-Caltech101 loader (not selected).")

    if "shd" in selected_datasets:
        shd_train_loader, shd_test_loader = get_shd_loaders(
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            save_to=str(args.data_dir),
        )
    else:
        logger.info("Skipping SHD loader (not selected).")
    logger.info("All dataset loaders are ready.")

    datasets = []
    if ncaltech_train_loader is not None and ncaltech_test_loader is not None:
        datasets.append(
            {
                "dataset_name": "N-Caltech101",
                "modality": "vision",
                "input_type": "dvs_gesture",
                "num_classes": 101,
                "train_loader": ncaltech_train_loader,
                "test_loader": ncaltech_test_loader,
                "spatial_size": vision_spatial,
                "input_size": None,
            }
        )
    if shd_train_loader is not None and shd_test_loader is not None:
        datasets.append(
            {
                "dataset_name": "SHD-20",
                "modality": "audio",
                "input_type": "shd",
                "num_classes": 20,
                "train_loader": shd_train_loader,
                "test_loader": shd_test_loader,
                "spatial_size": None,
                "input_size": 700,
            }
        )
    if not datasets:
        raise RuntimeError("No datasets selected. Use --datasets shd and/or ncaltech101.")
    configs = [
        {"config_name": "baseline_snn", "enable_structural_plasticity": False},
        {"config_name": "structural_plasticity", "enable_structural_plasticity": True},
    ]

    sweep_tasks = [(cfg, ds) for cfg in configs for ds in datasets]
    all_results: Dict[str, Dict[str, Any]] = {}
    summary_rows: List[Dict[str, Any]] = []

    for cfg, ds in tqdm(sweep_tasks, desc="Experiment Sweep", unit="run"):
        result = run_single_sweep(
            config_name=cfg["config_name"],
            dataset_name=ds["dataset_name"],
            modality=ds["modality"],
            input_type=ds["input_type"],
            num_classes=ds["num_classes"],
            train_loader=ds["train_loader"],
            test_loader=ds["test_loader"],
            device=device,
            args=args,
            results_dir=args.results_dir,
            logger=logger,
            spatial_size=ds["spatial_size"],
            input_size=ds["input_size"],
            enable_structural_plasticity=cfg["enable_structural_plasticity"],
        )
        key = f"{cfg['config_name']}::{ds['dataset_name']}"
        all_results[key] = result
        summary_rows.append(
            {
                "config_name": cfg["config_name"],
                "dataset_name": ds["dataset_name"],
                "final_specs": result["final_specs"],
            }
        )

    # Generate temporal engram evolution plots (Vision vs Audio) per configuration.
    for cfg in configs:
        cfg_name = cfg["config_name"]
        vision_key = f"{cfg_name}::N-Caltech101"
        audio_key = f"{cfg_name}::SHD-20"
        if vision_key not in all_results or audio_key not in all_results:
            continue
        plot_path = args.results_dir / f"{cfg_name}__engram_evolution.png"
        logger.info("Saving temporal engram plot: %s", plot_path)
        plot_engram_evolution(
            vision_results=all_results[vision_key],
            audio_results=all_results[audio_key],
            save_path=plot_path,
            title=f"Temporal Engram Evolution ({cfg_name})",
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{timestamp}__" if args.timestamped_output else ""
    aggregate_payload = {
        "meta": {
            "device": str(device),
            "num_runs": args.num_runs,
            "base_seed": args.base_seed,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "results_dir": str(args.results_dir),
        },
        "experiments": all_results,
    }
    aggregate_json_path = args.results_dir / f"{prefix}research_sweep_results.json"
    final_specs_json_path = args.results_dir / f"{prefix}final_specs.json"
    _save_json(aggregate_json_path, aggregate_payload)
    _save_json(final_specs_json_path, {k: v.get("final_specs", {}) for k, v in all_results.items()})

    md_table = _build_markdown_summary_table(summary_rows)
    logger.info("Sweep complete. Aggregate JSON: %s", aggregate_json_path)
    logger.info("Final specs JSON: %s", final_specs_json_path)

    print("\n## Research Sweep Summary\n")
    print(md_table)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
