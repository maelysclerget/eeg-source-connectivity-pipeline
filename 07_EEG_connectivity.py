#!/usr/bin/env python3
# Author Stavriani Skarvelaki / Maelys Clerget

"""
Compute ROI-to-ROI source-space EEG connectivity.

This script reads the ROI Epochs FIF files created by
08_EEG_epoch_label_time_courses.py and computes baseline-corrected
ROI x ROI imaginary coherence matrices.

For task, the expected inverse-solution layout is:

    inverse_solution/<mode>/<method>/task/cov<label>/

For RSpre and other non-task recordings, the expected layout is:

    inverse_solution/<mode>/<method>/<task>/

The input channels are ROIs.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from utils07 import (
    base_tag,
    chmod_group,
    compute_connectivity,
    load_roi_epochs,
    output_directory,
    save_connectivity_circle,
    save_heatmap,
    save_matrix_csv,
)


EEG_DERIVATIVES_DIR = Path("/work/uphummel/studies/tTIS-EEG/derivatives/EEG")

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "low_beta": (13.0, 20.0),
    "high_beta": (20.0, 30.0),
}

INVERSE_METHODS = ("MNE", "sLORETA", "eLORETA")
SOURCE_MODES = ("surface", "volume", "mixed")


def selected_frequency_bands(args: argparse.Namespace) -> list[str]:
    """Return requested frequency bands after checking their names."""
    selected_bands = args.bands if args.bands else list(BANDS)
    invalid_bands = [band for band in selected_bands if band not in BANDS]
    if invalid_bands:
        raise ValueError(f"Unknown band(s): {invalid_bands}. Available bands: {list(BANDS)}")

    return selected_bands


def append_summary_row(
    summary_rows: list[dict],
    band_name: str,
    fmin: float,
    fmax: float,
    roi_names: list[str],
    baseline_data: np.ndarray,
    nonbaseline_data: np.ndarray,
    matrix: np.ndarray,
    block: int | str,
) -> None:
    """Add one connectivity result to the output summary table."""
    nonzero = matrix[matrix != 0]
    summary_rows.append(
        {
            "Block": block,
            "Band": band_name,
            "Fmin": fmin,
            "Fmax": fmax,
            "N_ROIs": len(roi_names),
            "N_baseline_epochs": baseline_data.shape[0],
            "N_nonbaseline_epochs": nonbaseline_data.shape[0],
            "N_baseline_samples": baseline_data.shape[2],
            "N_nonbaseline_samples": nonbaseline_data.shape[2],
            "Min": float(np.min(matrix)),
            "Max": float(np.max(matrix)),
            "Mean": float(np.mean(nonzero)) if nonzero.size else 0.0,
            "STD": float(np.std(nonzero)) if nonzero.size else 0.0,
        }
    )


def process_connectivity(args: argparse.Namespace) -> None:
    """Compute ROI-level connectivity for every selected frequency band."""
    selected_bands = selected_frequency_bands(args)

    out_dir = output_directory(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out_dir, 0o2770)

    baseline_data, roi_names, sfreq, baseline_source = load_roi_epochs(args, "baseline")
    nonbaseline_data, nonbaseline_roi_names, nonbaseline_sfreq, nonbaseline_source = load_roi_epochs(args, "nonbaseline")
    if roi_names != nonbaseline_roi_names:
        raise ValueError("Baseline and nonbaseline ROI names do not match.")
    if not np.isclose(sfreq, nonbaseline_sfreq):
        raise ValueError(f"Baseline and nonbaseline sfreq differ: {sfreq} vs {nonbaseline_sfreq}")

    tag = base_tag(args)

    print(f"Subject: sub-{args.subject}")
    print(f"Session: ses-{args.session}")
    print(f"Task: task-{args.task}")
    print(f"Source mode: {args.mode}")
    print(f"Method: {args.method}")
    print(f"Connectivity method: {args.connectivity_method}")
    print(f"Baseline epochs: {baseline_source}")
    print(f"Nonbaseline epochs: {nonbaseline_source}")
    print(f"Number of retained ROIs: {len(roi_names)}")
    print(f"Baseline epochs: {baseline_data.shape[0]}")
    print(f"Nonbaseline epochs: {nonbaseline_data.shape[0]}")
    print(f"Sampling frequency: {sfreq:g} Hz")
    print(f"Output: {out_dir}")

    summary_rows = []

    if args.task.lower() == "task":
        process_task_blocks(
            args,
            selected_bands,
            baseline_data,
            nonbaseline_data,
            roi_names,
            sfreq,
            out_dir,
            tag,
            summary_rows,
        )
    else:
        process_resting_recording(
            args,
            selected_bands,
            baseline_data,
            nonbaseline_data,
            roi_names,
            sfreq,
            out_dir,
            tag,
            summary_rows,
        )

    summary_path = out_dir / f"{tag}_connectivity_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    chmod_group(summary_path)
    print(f"Saved: {summary_path}")
    print("Finished.")


def process_resting_recording(
    args: argparse.Namespace,
    selected_bands: list[str],
    baseline_data: np.ndarray,
    nonbaseline_data: np.ndarray,
    roi_names: list[str],
    sfreq: float,
    out_dir: Path,
    tag: str,
    summary_rows: list[dict],
) -> None:
    """Baseline-correct one non-task recording against its first baseline window."""
    for band_name in selected_bands:
        fmin, fmax = BANDS[band_name]
        print(f"Computing {band_name}: {fmin:g}-{fmax:g} Hz")

        baseline_matrix = compute_connectivity(baseline_data, sfreq, fmin, fmax, args.connectivity_method)
        nonbaseline_matrix = compute_connectivity(nonbaseline_data, sfreq, fmin, fmax, args.connectivity_method)
        matrix = nonbaseline_matrix - baseline_matrix
        file_tag = f"{tag}_band-{band_name}"
        matrix_csv = out_dir / f"{file_tag}_connectivity_matrix.csv"
        heatmap_png = out_dir / f"{file_tag}_connectivity_heatmap.png"
        circle_png = out_dir / f"{file_tag}_connectivity_circle.png"

        save_matrix_csv(matrix, roi_names, matrix_csv)
        save_heatmap(matrix,
            roi_names,
            heatmap_png,
            f"{args.task} ROI connectivity - {args.method} - {band_name} ({fmin:g}-{fmax:g} Hz)",
        )
        save_connectivity_circle(
            matrix,
            roi_names,
            circle_png,
            f"{args.task} ROI connectivity circle - {args.method} - {band_name} ({fmin:g}-{fmax:g} Hz)",
        )

        append_summary_row(summary_rows, band_name, fmin, fmax, roi_names, baseline_data, nonbaseline_data, matrix, "all")

        print(f"Saved: {matrix_csv}")
        print(f"Saved: {heatmap_png}")
        print(f"Saved: {circle_png}")


def process_task_blocks(
    args: argparse.Namespace,
    selected_bands: list[str],
    baseline_data: np.ndarray,
    nonbaseline_data: np.ndarray,
    roi_names: list[str],
    sfreq: float,
    out_dir: Path,
    tag: str,
    summary_rows: list[dict],
) -> None:
    """Baseline-correct each task block using the baseline epoch from the same S15 block."""
    n_blocks = baseline_data.shape[0] #n_baseline_epochs x n_ROIs x n_times
    expected_nonbaseline_epochs = n_blocks * args.n_epochs_per_block #9 blocks and 19 nonbaseline epochs per block = 171
    if nonbaseline_data.shape[0] != expected_nonbaseline_epochs:
        raise ValueError(
            "Task nonbaseline epochs do not match the expected block layout: "
            f"{nonbaseline_data.shape[0]} epochs found, expected "
            f"{n_blocks} baseline blocks x {args.n_epochs_per_block} epochs/block = "
            f"{expected_nonbaseline_epochs}."
        )

    print(f"Task blocks: {n_blocks}")
    print(f"Nonbaseline epochs per block: {args.n_epochs_per_block}")

    for band_name in selected_bands:
        fmin, fmax = BANDS[band_name]
        print(f"Computing {band_name}: {fmin:g}-{fmax:g} Hz")

        for block_index in range(n_blocks):
            block_number = block_index + 1
            start = block_index * args.n_epochs_per_block
            stop = start + args.n_epochs_per_block
            block_baseline_data = baseline_data[block_index : block_index + 1]
            block_nonbaseline_data = nonbaseline_data[start:stop] #block 1: nonbaseline_data[0:19] block 2: nonbaseline_data[19:38] block 3: nonbaseline_data[38:57]

            baseline_matrix = compute_connectivity(block_baseline_data, sfreq, fmin, fmax, args.connectivity_method)
            nonbaseline_matrix = compute_connectivity(block_nonbaseline_data, sfreq, fmin, fmax, args.connectivity_method)
            matrix = nonbaseline_matrix - baseline_matrix

            file_tag = f"{tag}_block-{block_number:02d}_band-{band_name}"
            matrix_csv = out_dir / f"{file_tag}_connectivity_matrix.csv"
            heatmap_png = out_dir / f"{file_tag}_connectivity_heatmap.png"
            circle_png = out_dir / f"{file_tag}_connectivity_circle.png"

            save_matrix_csv(matrix, roi_names, matrix_csv)
            save_heatmap(
                matrix,
                roi_names,
                heatmap_png,
                f"task block {block_number} ROI connectivity - {args.method} - {band_name} ({fmin:g}-{fmax:g} Hz)",
            )
            save_connectivity_circle(
                matrix,
                roi_names,
                circle_png,
                f"task block {block_number} ROI connectivity circle - {args.method} - {band_name} ({fmin:g}-{fmax:g} Hz)",
            )

            append_summary_row(
                summary_rows,
                band_name,
                fmin,
                fmax,
                roi_names,
                block_baseline_data,
                block_nonbaseline_data,
                matrix,
                block_number,
            )

            print(f"Saved: {matrix_csv}")
            print(f"Saved: {heatmap_png}")
            print(f"Saved: {circle_png}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute ROI-to-ROI source-space imaginary coherence."
    )
    parser.add_argument("--subject", required=True, help="Subject ID without 'sub-', for example 41Y01.")
    parser.add_argument("--session", required=True, help="Session without 'ses-', for example 1.")
    parser.add_argument("--task", default="RSpre", help="Task to process, for example RSpre or task.")
    parser.add_argument("--mode", default="surface", choices=SOURCE_MODES, help="Source-space mode.")
    parser.add_argument("--method", default="MNE", choices=INVERSE_METHODS, help="Inverse method.")
    parser.add_argument("--cov-label", default="15", help="Covariance label for task outputs. Default: 15.")
    parser.add_argument("--derivatives-dir", default=str(EEG_DERIVATIVES_DIR), help="EEG derivatives root.")
    parser.add_argument("--output-dir", default=None, help="Optional explicit output directory.")
    parser.add_argument("--bands", nargs="+", choices=list(BANDS), default=None, help="Bands to compute. Default: all.")
    parser.add_argument("--connectivity-method", default="imcoh", help="Connectivity method. Default: imcoh.")
    parser.add_argument("--n-epochs-per-block", type=int, default=19, help="Task nonbaseline epochs per S15 block. Default: 19.")
    return parser.parse_args()


if __name__ == "__main__":
    process_connectivity(parse_args())
