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

from utils08 import (
    labels_evoked_to_raw,
    make_rspre_epochs,
    make_task_epochs,
    print_annotation_descriptions,
    save_epochs,
)


DERIVATIVES_DIR = "/work/uphummel/studies/tTIS-EEG/derivatives/EEG"

# Defaults used when running:
#     python 08_EEG_epoch_label_time_courses.py
subject = "41Y01"
session = "1"
task = "RSpre"
mode = "volume"
method = "MNE"
cov_label = "15"
labels_fif = None


def chmod_group(path: str | Path) -> None:
    """Keep cluster outputs group-readable/writable when possible."""
    try:
        os.chmod(path, 0o770)
    except PermissionError:
        print(f"Could not chmod {path}; continuing.")


def get_raw_fif_path(args: argparse.Namespace) -> Path:
    """Return the preprocessed raw FIF path that contains task annotations."""
    raw_path = (
        Path(args.derivatives_dir)
        / f"sub-{args.subject}"
        / f"ses-{args.session}"
        / args.task
        / (
            f"sub-{args.subject}_ses-{args.session}_task-{args.task}"
            "_eeg_not_interpolated_final_preprocessed_eeg.fif"
        )
    )

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw FIF not found for annotations: {raw_path}")

    return raw_path


def default_labels_fif_path(args: argparse.Namespace) -> Path:
    """Return the default labels FIF path from the step 06 output layout."""
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
        if not args.cov_label:
            raise ValueError("task epoching needs --cov-label, e.g. 15.")
        cov_tag = args.cov_label if str(args.cov_label).startswith("s") else f"s{args.cov_label}"
        return inverse_dir / f"cov-{cov_tag}" / f"{base_tag}_cov-{cov_tag}_labels.fif"

    return inverse_dir / args.task / f"{base_tag}_labels.fif"


def save_annotation_sidecar(raw_path: Path, labels_path: Path, out_dir: Path) -> mne.Annotations:
    """Load raw annotations, print them, and save them next to labels epochs."""
    print("\nOpening raw FIF for annotations:")
    print(raw_path)
    raw = mne.io.read_raw_fif(raw_path, preload=False, verbose="ERROR")
    annotations = raw.annotations
    print_annotation_descriptions(annotations)

    sidecar_path = out_dir / f"{labels_path.stem}_annot.fif"
    annotations.save(sidecar_path, overwrite=True)
    chmod_group(sidecar_path)
    print(f"Saved annotation sidecar: {sidecar_path}")

    return annotations


def process_label_epochs(args: argparse.Namespace) -> None:
    labels_path = Path(args.labels_fif) if args.labels_fif else default_labels_fif_path(args)
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels FIF not found: {labels_path}")

    out_dir = Path(args.output_dir) if args.output_dir else labels_path.parent / "epochs"
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
    annotations = None
    if task_lower == "task":
        raw_path = get_raw_fif_path(args)
        annotations = save_annotation_sidecar(raw_path, labels_path, out_dir)
    else:
        print(f"\n{args.task} selected: annotations are not needed for epoching.")

    raw_labels = labels_evoked_to_raw(evoked, annotations=annotations)

    if task_lower == "task":
        baseline_no_stim_epochs, baseline_stim_epochs, nonbaseline_epochs = make_task_epochs(
            raw_labels,
            epoch_duration=args.epoch_duration,
            n_epochs_per_block=args.n_epochs_per_block,
            s15_annotation=args.s15_annotation,
            block_start_annotation=args.block_start_annotation,
            block_end_annotation=args.block_end_annotation,
        )
    else:
        baseline_epochs, nonbaseline_epochs = make_rspre_epochs(
            raw_labels,
            baseline_duration=args.rspre_baseline_duration,
            epoch_duration=args.epoch_duration,
        )

    stem = labels_path.stem
    baseline_path = out_dir / f"{stem}_baseline-epo.fif"
    nonbaseline_path = out_dir / f"{stem}_nonbaseline-epo.fif"
    if task_lower == "task":
        baseline_no_stim_path = out_dir / f"{stem}_baseline_no_stim-epo.fif"
        baseline_stim_path = out_dir / f"{stem}_baseline_stim-epo.fif"
        save_epochs(baseline_no_stim_epochs, baseline_no_stim_path)
        save_epochs(baseline_stim_epochs, baseline_stim_path)
        baseline_epochs = baseline_stim_epochs if args.connectivity_baseline == "stim" else baseline_no_stim_epochs
        print(f"Saving {args.connectivity_baseline} baseline as compatibility file: {baseline_path}")

    save_epochs(baseline_epochs, baseline_path)
    save_epochs(nonbaseline_epochs, nonbaseline_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Epoch ROI labels FIF files for task or RSpre source connectivity."
    )
    parser.add_argument("--labels-fif", default=labels_fif, help="Input ROI labels FIF saved as EvokedArray.")
    parser.add_argument("--subject", default=subject, help="Subject ID without sub- prefix, e.g. 41Y01.")
    parser.add_argument("--session", default=session, help="Session ID without ses- prefix, e.g. 1.")
    parser.add_argument("--task", default=task, help="Task to epoch.")
    parser.add_argument("--mode", default=mode, help="Source space mode used for the labels FIF.")
    parser.add_argument("--method", default=method, help="Inverse method used for the labels FIF.")
    parser.add_argument("--cov-label", default=cov_label, help="Task covariance label, e.g. 15.")
    parser.add_argument("--derivatives-dir", default=DERIVATIVES_DIR, help="EEG derivatives root.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory. Default: labels_fif/../epochs.")
    parser.add_argument("--s15-annotation", default="Stimulus/S 15", help="Task S15 annotation used for S15-20s baseline and S15-to-S10 baseline.")
    parser.add_argument("--block-start-annotation", default="Stimulus/S 10", help="Task annotation used as block start for non-baseline epochs.")
    parser.add_argument("--block-end-annotation", default="Stimulus/S  8", help="Task annotation used as block end for non-baseline epochs.")
    parser.add_argument("--epoch-duration", type=float, default=5.0, help="Non-baseline epoch duration in seconds.")
    parser.add_argument("--connectivity-baseline", choices=("stim", "no_stim"), default="stim", help="For task data, which baseline to also save as *_baseline-epo.fif for connectivity compatibility.")
    parser.add_argument("--n-epochs-per-block", type=int, default=None, help="For task data, optional maximum number of 5 s epochs kept from each S10-to-S8 block. Default: keep all complete epochs in each block.")
    parser.add_argument("--rspre-baseline-duration", type=float, default=20.0, help="For RSpre, duration in seconds of the first-recording baseline epoch.")
    return parser.parse_args()


if __name__ == "__main__":
    process_label_epochs(parse_args())
