#!/usr/bin/env python3
# Author Stavriani Skarvelaki / Maelys Clerget

"""
Epoch ROI label time courses saved by 06_EEG_inverse_solution.py.

The input labels FIF stays an EvokedArray on disk. For epoching only, this
script wraps its data in a temporary RawArray, attaches annotations loaded from
the corresponding preprocessed raw FIF, and writes Epochs FIF outputs.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import mne
import pandas as pd

from utils08 import (
    labels_evoked_to_raw,
    make_rs_epochs,
    make_task_epochs,
    print_annotation_descriptions,
    save_epochs,
)


DERIVATIVES_DIR = "/work/uphummel/studies/tTIS-EEG/derivatives/EEG"


def chmod_group(path: str | Path) -> None:
    """Keep cluster outputs group-readable/writable when possible."""
    try:
        os.chmod(path, 0o770)
    except PermissionError:
        print(f"Could not chmod {path}; continuing.")


def get_raw_fif_path(args: argparse.Namespace) -> Path:
    """Return the preprocessed raw FIF path that contains annotations."""
    stem = f"sub-{args.subject}_ses-{args.session}_task-{args.task}"
    filename = f"{stem}_eeg_not_interpolated_final_preprocessed_eeg.fif"

    # This raw EEG file is used only to get annotations, not for source data.
    raw_path = (
        Path(args.derivatives_dir)
        / f"sub-{args.subject}"
        / f"ses-{args.session}"
        / args.task
        / filename
    )

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw FIF not found for annotations: {raw_path}")

    return raw_path


def default_labels_fif_path(args: argparse.Namespace) -> Path:
    """Return the default labels FIF path from the step 06 output layout."""
    # This labels FIF is the source/ROI time course file that will be epoched.
    inverse_dir = (
        Path(args.derivatives_dir)
        / f"sub-{args.subject}"
        / f"ses-{args.session}"
        / "source_reconstruction"
        / "inverse_solution"
        / args.mode
        / args.method
    )

    base_tag = f"{args.subject}_ses{args.session}_task-{args.task}_src-{args.mode}_method-{args.method}"
    if args.task.lower() == "task":
        cov_tag = args.cov_label if str(args.cov_label).startswith("s") else f"s{args.cov_label}"
        return inverse_dir / "task" / f"cov{cov_tag}" / f"{base_tag}_cov-{cov_tag}_labels.fif"

    return inverse_dir / args.task / f"{base_tag}_labels.fif"


def default_output_dir(args: argparse.Namespace) -> Path:
    """Return the new epoch output directory in the subject/session task folder."""
    # Epoch files are saved in the task folder
    return (
        Path(args.derivatives_dir)
        / f"sub-{args.subject}"
        / f"ses-{args.session}"
        / args.task
        / "epochs"
        / args.mode
        / args.method
    )


def epoch_output_stem(args: argparse.Namespace) -> str:
    """Return the epoch filename stem."""
    return f"{args.subject}_ses{args.session}_task-{args.task}_src-{args.mode}_method-{args.method}"


def save_annotation_sidecar(raw_path: Path, out_dir: Path, stem: str) -> mne.Annotations:
    """Load raw annotations, print them, and save them next to labels epochs."""
    print("\nOpening raw FIF for annotations:")
    print(raw_path)
    raw = mne.io.read_raw_fif(raw_path, preload=False, verbose="ERROR") 
    annotations = raw.annotations # Reads annotations of raw preprocessed EEG
    print_annotation_descriptions(annotations)

    # Save the raw annotations next to the epochs for checking/debugging.
    sidecar_path = out_dir / f"{stem}_annot.fif"
    annotations.save(sidecar_path, overwrite=True)
    chmod_group(sidecar_path)
    print(f"Saved annotation sidecar: {sidecar_path}")

    return annotations


def save_sequence_epochs(sequence_epochs, out_dir: Path, stem: str) -> None:
    """Save variable-length sequence epochs as one FIF per sequence interval."""
    rows = []
    for block, sequence, duration, epochs in sequence_epochs:
        # Sequence epochs can have different durations, so each is saved separately.
        path = out_dir / f"{stem}_task_sequence_block-{block:02d}_seq-{sequence:02d}-epo.fif"
        save_epochs(epochs, path)
        chmod_group(path)
        rows.append(
            {
                "block": block,
                "sequence": sequence,
                "duration_s": duration,
                "path": str(path),
            }
        )

    # Index file lists all separate sequence epoch files and their durations.
    index_path = out_dir / f"{stem}_task_sequence_index.csv"
    pd.DataFrame(rows).to_csv(index_path, index=False)
    chmod_group(index_path)
    print(f"Saved sequence index: {index_path}")


def process_label_epochs(args: argparse.Namespace) -> None:
    labels_path = Path(args.labels_fif) if args.labels_fif else default_labels_fif_path(args)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels FIF not found: {labels_path}")

    out_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(out_dir, 0o2770)

    print("\nOpening labels FIF:")
    print(labels_path)
    evoked = mne.read_evokeds(labels_path, condition=0, verbose="ERROR")
    print(evoked)
    print(f"Labels/ROIs: {len(evoked.ch_names)}")
    print(f"Sampling frequency: {evoked.info['sfreq']:g} Hz")
    print(f"Time range: {evoked.times[0]:.3f}-{evoked.times[-1]:.3f} s")
    if evoked.tmin != 0:
        raise ValueError(f"Expected labels FIF to start at 0 s, got tmin={evoked.tmin}.")

    task_lower = args.task.lower()
    raw_path = get_raw_fif_path(args)
    stem = epoch_output_stem(args)
    annotations = save_annotation_sidecar(raw_path, out_dir, stem)

    # Convert continuous ROI time courses to RawArray so MNE can epoch them.
    raw_labels = labels_evoked_to_raw(evoked, annotations=annotations)

    if task_lower == "task":
        # Task creates baseline epochs, fixed 5 s task epochs, and sequence epochs.
        baseline_no_stim_epochs, baseline_stim_epochs, fixed_task_epochs, sequence_epochs = make_task_epochs(
            raw_labels,
            epoch_duration=args.epoch_duration,
            s15_annotation=args.s15_annotation,
            block_start_annotation=args.block_start_annotation,
        )
        baseline_no_stim_path = out_dir / f"{stem}_baseline_no_stim-epo.fif"
        baseline_stim_path = out_dir / f"{stem}_baseline_stim-epo.fif"
        task_fixed_path = out_dir / f"{stem}_task_fixed-epo.fif"

        save_epochs(baseline_no_stim_epochs, baseline_no_stim_path)
        save_epochs(baseline_stim_epochs, baseline_stim_path)
        save_epochs(fixed_task_epochs, task_fixed_path)
        save_sequence_epochs(sequence_epochs, out_dir, stem)
        return

    # RSpre/RSpost/RSstim creates fixed 5 s epochs after S15.
    rs_epochs = make_rs_epochs(
        raw_labels,
        task_name=args.task,
        epoch_duration=args.epoch_duration,
        s15_annotation=args.s15_annotation,
    )
    rs_path = out_dir / f"{stem}_rs_s15_epochs-epo.fif"
    save_epochs(rs_epochs, rs_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Epoch ROI labels FIF files for task or RSpre source connectivity."
    )
    parser.add_argument("--labels-fif", default=None, help="Input ROI labels FIF saved as EvokedArray.")
    parser.add_argument("--subject", default="41Y01", help="Subject ID without sub- prefix, e.g. 41Y01.")
    parser.add_argument("--session", default="1", help="Session ID without ses- prefix, e.g. 1.")
    parser.add_argument("--task", default="task", help="Task to epoch.")
    parser.add_argument("--mode", default="surface", help="Source space mode used for the labels FIF.")
    parser.add_argument("--method", default="MNE", help="Inverse method used for the labels FIF.")
    parser.add_argument("--cov-label", default="15", help="Task covariance label, e.g. 15.")
    parser.add_argument("--derivatives-dir", default=DERIVATIVES_DIR, help="EEG derivatives root.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory. Default: task/epochs/mode/method.")
    parser.add_argument("--s15-annotation", default="Stimulus/S 15", help="S15 annotation used for RS blocks and task baselines.")
    parser.add_argument("--block-start-annotation", default="Stimulus/S 10", help="Task S10 annotation used as fixed and sequence block start.")
    parser.add_argument("--epoch-duration", type=float, default=5.0, help="Epoch duration in seconds for fixed epochs.")
    return parser.parse_args()


if __name__ == "__main__":
    process_label_epochs(parse_args())
