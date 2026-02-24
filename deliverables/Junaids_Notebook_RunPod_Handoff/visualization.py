"""
Visualization helpers for temporal engram analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


SNAPSHOT_ORDER = ("t25", "t50", "t75", "t100")
SNAPSHOT_LABELS = {
    "t25": "25%",
    "t50": "50%",
    "t75": "75%",
    "t100": "100%",
}


def _extract_stats_map(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract temporal silhouette summary from a run_seed_iterator result or summary dict.
    """
    if not isinstance(results, dict):
        raise ValueError("Expected a dictionary for temporal silhouette results.")

    # Preferred shape from training.pipeline.run_seed_iterator(...)
    temporal_summary = results.get("temporal_silhouette_summary")
    if isinstance(temporal_summary, dict) and temporal_summary:
        # If nested by prefix (e.g., {"vision": {...}}), take the first nested dict.
        if all(isinstance(v, dict) and "mean" not in v for v in temporal_summary.values()):
            first_key = next(iter(temporal_summary))
            candidate = temporal_summary[first_key]
            if isinstance(candidate, dict):
                return candidate
        return temporal_summary

    # Fallback: flat summary map with keys like "temporal_silhouette.t25"
    flat_summary = results.get("summary", results)
    if not isinstance(flat_summary, dict):
        raise ValueError("Could not find a valid summary dictionary.")

    extracted: Dict[str, Any] = {}
    for key, value in flat_summary.items():
        if key.startswith("temporal_silhouette."):
            snap = key.split("temporal_silhouette.", 1)[1]
            extracted[snap] = value
    if extracted:
        return extracted

    # Last fallback: already provided as {"t25": {...}, ...}
    if any(k in flat_summary for k in SNAPSHOT_ORDER):
        return flat_summary

    raise ValueError("No temporal silhouette metrics found in the provided results.")


def _extract_curve(results: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    stats_map = _extract_stats_map(results)
    means = []
    stds = []
    for snap in SNAPSHOT_ORDER:
        item = stats_map.get(snap, {})
        if isinstance(item, dict):
            means.append(float(item.get("mean", np.nan)))
            stds.append(float(item.get("std_dev", item.get("std", 0.0))))
        else:
            means.append(float(item))
            stds.append(0.0)
    return np.asarray(means, dtype=float), np.asarray(stds, dtype=float)


def plot_engram_evolution(
    vision_results: Dict[str, Any],
    audio_results: Dict[str, Any],
    save_path: Optional[Path] = None,
    title: str = "Temporal Engram Evolution",
) -> None:
    """
    Plot temporal silhouette score evolution for vision vs audio experiments.

    Args:
        vision_results: `run_seed_iterator(...)` output (or compatible summary dict)
            for the vision experiment (e.g., N-Caltech/N-MNIST).
        audio_results: `run_seed_iterator(...)` output (or compatible summary dict)
            for the audio experiment (e.g., SHD).
        save_path: Optional path to save the figure.
        title: Plot title.
    """
    x = np.arange(len(SNAPSHOT_ORDER))
    xticklabels = [SNAPSHOT_LABELS[s] for s in SNAPSHOT_ORDER]

    vision_mean, vision_std = _extract_curve(vision_results)
    audio_mean, audio_std = _extract_curve(audio_results)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    ax.plot(x, vision_mean, marker="o", linewidth=2.2, color="#1f77b4", label="Vision (N-Caltech)")
    ax.plot(x, audio_mean, marker="s", linewidth=2.2, color="#d62728", label="Audio (SHD)")

    if np.isfinite(vision_std).any():
        ax.fill_between(x, vision_mean - vision_std, vision_mean + vision_std, color="#1f77b4", alpha=0.15)
    if np.isfinite(audio_std).any():
        ax.fill_between(x, audio_mean - audio_std, audio_mean + audio_std, color="#d62728", alpha=0.15)

    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("Time-step Snapshot")
    ax.set_ylabel("Silhouette Score")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

