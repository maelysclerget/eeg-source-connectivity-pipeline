"""Utility functions for source-space connectivity scripts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne_connectivity import spectral_connectivity_epochs
try:
    from mne_connectivity.viz import plot_connectivity_circle
except ImportError:
    from mne.viz import plot_connectivity_circle


def chmod_group(path: str | Path) -> None:
    try:
        os.chmod(path, 0o770)
    except PermissionError:
        print(f"Could not chmod {path}; continuing.")


def epoch_base_tag(args: argparse.Namespace) -> str:
    """Return the filename stem used by step 08 epoch outputs."""
    return f"{args.subject}_ses{args.session}_task-{args.task}_src-{args.mode}_method-{args.method}"


def connectivity_tag(args: argparse.Namespace) -> str:
    """Return the filename stem used for connectivity outputs."""
    tag = epoch_base_tag(args)
    baseline_kind = getattr(args, "baseline_kind", "baseline")
    if args.task.lower() == "task" and baseline_kind != "baseline":
        baseline_label = baseline_kind.replace("baseline_", "")
        tag = f"{tag}_baseline-{baseline_label}"

    return tag


def output_directory(args: argparse.Namespace) -> Path:
    """Return the connectivity output directory."""
    if args.output_dir:
        return Path(args.output_dir)

    return (
        Path(args.derivatives_dir)
        / f"sub-{args.subject}"
        / f"ses-{args.session}"
        / "connectivity"
        / args.mode
        / args.method
        / args.task
    )


def read_roi_epochs(path: Path) -> tuple[np.ndarray, list[str], float]:
    """Read one ROI Epochs FIF file created by 08_EEG_epoch_label_time_courses.py."""
    if not path.exists():
        raise FileNotFoundError(f"ROI epochs FIF not found: {path}")

    epochs = mne.read_epochs(path, preload=True, verbose="ERROR")
    data = epochs.get_data()
    roi_names = list(epochs.ch_names)
    sfreq = float(epochs.info["sfreq"])

    if data.ndim != 3:
        raise ValueError(f"Expected 3D epochs data in {path}, got shape {data.shape}.")
    if len(roi_names) != data.shape[1]:
        raise ValueError(
            f"ROI name count does not match data channels in {path}: "
            f"{len(roi_names)} names for {data.shape[1]} channels."
        )

    return data, roi_names, sfreq


def roi_epoch_paths(args: argparse.Namespace, kind: str) -> list[Path]:
    """Return baseline or nonbaseline epoch files from the step 08 output layout."""
    directory = (
        Path(args.derivatives_dir)
        / f"sub-{args.subject}"
        / f"ses-{args.session}"
        / args.task
        / "epochs"
        / args.mode
        / args.method
    )
    tag = epoch_base_tag(args)

    if args.mode in ("surface", "volume"):
        return [directory / f"{tag}_labels_{kind}-epo.fif"]

    return [directory / f"{tag}_surface_labels_{kind}-epo.fif", directory / f"{tag}_volume_labels_{kind}-epo.fif"]


def load_roi_epochs(args: argparse.Namespace, kind: str) -> tuple[np.ndarray, list[str], float, str]:
    """Load baseline or nonbaseline ROI epochs, stacking surface/volume channels if needed."""
    loaded = []
    for path in roi_epoch_paths(args, kind):
        print(f"Reading {kind} ROI epochs: {path}")
        # each item in loaded is (data, roi_names, sfreq, path)
        loaded.append((*read_roi_epochs(path), str(path)))

    # sfreqs should match across surface and volume files in mixed mode.
    sfreqs = [item[2] for item in loaded]
    first_sfreq = sfreqs[0]
    if not all(np.isclose(first_sfreq, sfreq) for sfreq in sfreqs):
        raise ValueError(f"ROI epochs have different sampling frequencies: {sfreqs}")

    # data shape is n_epochs x n_ROIs x n_times.
    n_epochs = [item[0].shape[0] for item in loaded]
    n_times = [item[0].shape[2] for item in loaded]
    if len(set(n_epochs)) != 1 or len(set(n_times)) != 1:
        raise ValueError(
            f"ROI epochs do not align for {kind}: "
            f"n_epochs={n_epochs}, n_times={n_times}"
        )

    # For mixed mode, stack surface and volume ROIs on the channel/ROI axis.
    data = np.concatenate([item[0] for item in loaded], axis=1)
    roi_names = []
    for item in loaded:
        # Combines ROI names in the same order as the data channels.
        roi_names.extend(item[1])

    if data.shape[1] < 2:
        raise ValueError("Need at least two ROIs to compute connectivity.")
    if data.shape[2] < 2:
        raise ValueError("Need at least two time samples to compute connectivity.")

    source_text = " + ".join(item[3] for item in loaded)
    print(f"{kind.capitalize()} epochs shape: {data.shape}")

    return data, roi_names, first_sfreq, source_text


def compute_connectivity(roi_data: np.ndarray, sfreq: float, fmin: float, fmax: float, method: str) -> np.ndarray:
    """Compute all-to-all spectral connectivity between ROI time series."""
    if roi_data.ndim != 3:
        raise ValueError(f"Expected epochs as n_epochs x n_ROIs x n_times, got shape {roi_data.shape}.")

    con = spectral_connectivity_epochs(
        roi_data,
        method=method,
        mode="multitaper",
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        verbose=False,
    )

    matrix = con.get_data(output="dense")[:, :, 0]
    return np.abs(matrix)


def save_matrix_csv(matrix: np.ndarray, roi_names: list[str], path: Path) -> None:
    """Save an ROI x ROI connectivity matrix."""
    dataframe = pd.DataFrame(matrix, index=roi_names, columns=roi_names)
    dataframe.index.name = "ROI"
    dataframe.to_csv(path)
    chmod_group(path)


def save_heatmap(matrix: np.ndarray, roi_names: list[str], path: Path, title: str) -> None:
    """Save the lower triangle of the ROI connectivity matrix."""
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=0)
    figure_size = max(12, min(26, len(roi_names) * 0.30))
    fig, ax = plt.subplots(figsize=(figure_size, figure_size))

    masked_matrix = np.ma.array(matrix, mask=mask)
    image = ax.imshow(masked_matrix, cmap="coolwarm", aspect="equal")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Imaginary coherence")

    ax.set_xticks(np.arange(len(roi_names)))
    ax.set_yticks(np.arange(len(roi_names)))
    ax.set_xticklabels(roi_names)
    ax.set_yticklabels(roi_names)

    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", rotation=90, labelsize=6)
    ax.tick_params(axis="y", rotation=0, labelsize=6)
    ax.set_title(title, pad=60)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    chmod_group(path)

def save_connectivity_circle(matrix: np.ndarray, roi_names: list[str], path: Path, title: str, n_lines: int = 50) -> None:
    """Save a circular graph of the strongest ROI-to-ROI connections."""
    circle_matrix = matrix.copy()
    np.fill_diagonal(circle_matrix, 0.0)
    max_connections = len(roi_names) * (len(roi_names) - 1) // 2
    n_lines = min(n_lines, max_connections)

    fig, _ = plot_connectivity_circle(
        circle_matrix,
        node_names=roi_names,
        n_lines=n_lines,
        title=title,
        colormap="hot",
        facecolor="black",
        textcolor="white",
        fontsize_names=6,
        fontsize_title=12,
        show=False,
        interactive=False,
    )
    fig.savefig(path, dpi=300, facecolor="black", bbox_inches="tight")
    plt.close(fig)
    chmod_group(path)
