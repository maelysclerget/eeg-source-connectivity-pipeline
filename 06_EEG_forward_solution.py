#!/usr/bin/env python3
# Author Stavriani Skarvelaki / Maelys Clerget
"""
Compute reusable EEG forward solutions for one subject/session.

Outputs are session-level and method-independent:
derivatives/EEG/sub-*/ses-*/source_reconstruction/forward_solution/{surface,volume,mixed}
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np

from utils06 import (
    create_bem_solution,
    create_forward_solution,
    create_mixed_source_space,
    create_striatum_volume_source_space,
    create_surface_source_space,
    orient_forward_solution,
    save_dipole_area_index,
)

DERIVATIVES_DIR = "/work/uphummel/studies/tTIS-EEG/derivatives/EEG"
FREESURFER_DIR = "/work/uphummel/studies/tTIS-EEG/derivatives/MRI/freesurfer"
SOURCE_SPACE_MODES = ("surface", "volume", "mixed")


def chmod_group(path: str | Path) -> None:
    """Keep cluster outputs group-readable/writable when possible."""
    try:
        os.chmod(path, 0o770)
    except PermissionError:
        print(f"Could not chmod {path}; continuing.")


def chmod_group_dir(path: str | Path) -> None:
    """Keep cluster output directories group-readable/writable/setgid when possible."""
    try:
        os.chmod(path, 0o2770)
    except PermissionError:
        print(f"Could not chmod {path}; continuing.")


def freesurfer_subject(subject: str) -> str:
    return f"sub-{subject}_ses-baseline"


def get_personalized_montage_path(subject: str, session: str) -> Path:
    return (
        Path(DERIVATIVES_DIR)
        / f"sub-{subject}"
        / f"ses-{session}"
        / "personalized_montage"
        / f"{subject}_ses{session}_personalized_montage.fif"
    )


def get_trans_path(subject: str, session: str) -> Path:
    trans_path = (
        Path(DERIVATIVES_DIR)
        / f"sub-{subject}"
        / f"ses-{session}"
        / "source_reconstruction"
        / f"{subject}_ses{session}_trans.fif"
    )

    if not trans_path.exists():
        raise FileNotFoundError(f"Coregistration transform not found: {trans_path}")

    return trans_path


def get_forward_output_dir(subject: str, session: str, mode: str) -> Path:
    return (
        Path(DERIVATIVES_DIR)
        / f"sub-{subject}"
        / f"ses-{session}"
        / "source_reconstruction"
        / "forward_solution"
        / mode
    )


def get_forward_base_tag(subject: str, session: str, mode: str) -> str:
    return f"{subject}_ses{session}_src-{mode}"


def make_source_space(mode: str, subject_fs: str, subjects_dir: Path, bem, spacing: str, pos: float):
    if mode == "surface":
        return create_surface_source_space(subject_fs, subjects_dir, spacing=spacing)
    if mode == "volume":
        return create_striatum_volume_source_space(subject_fs, subjects_dir, bem=bem, pos=pos)
    if mode == "mixed":
        return create_mixed_source_space(subject_fs, subjects_dir, bem=bem, spacing=spacing, pos=pos)
    raise ValueError("mode must be 'surface', 'volume', or 'mixed'")


def plot_forward_maps(fwd, mode: str, plot_dir: Path, base_tag: str) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    fwd_oriented = orient_forward_solution(fwd, mode)
    leadfield = fwd_oriented["sol"]["data"]
    eeg_names = list(fwd_oriented["sol"]["row_names"])

    print("Saving leadfield matrix plots.")
    block_size = 500
    n_dipoles = leadfield.shape[1]
    for start in range(0, n_dipoles, block_size):
        end = min(start + block_size, n_dipoles)
        last = end - 1
        fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
        im = ax.imshow(
            leadfield[:, start:end],
            aspect="auto",
            cmap="RdBu_r",
            extent=[start - 0.5, last + 0.5, len(eeg_names) - 0.5, -0.5],
        )
        ax.set_title(f"Leadfield - dipoles {start} to {last} - Mode: {mode}")
        ax.set_xlabel("Dipole index")
        ax.set_ylabel("EEG channel")
        x_ticks = list(range(start, last + 1, 100))
        if x_ticks[-1] != last:
            x_ticks.append(last)
        ax.set_xticks(x_ticks)
        ax.set_xlim(start - 0.5, last + 0.5)
        ax.set_yticks(np.arange(len(eeg_names)))
        ax.set_yticklabels(eeg_names, fontsize=6)
        fig.colorbar(im, ax=ax)
        fig.savefig(plot_dir / f"{base_tag}_leadfield_{start}_{last}.png", dpi=300)
        plt.close(fig)

    eeg_map = mne.sensitivity_map(fwd_oriented, ch_type="eeg", mode="fixed")
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.hist(eeg_map.data.ravel(), bins=50, color="c")
    ax.set_title(f"EEG Sensitivity - {mode}")
    ax.set_xlabel("EEG sensitivity value")
    ax.set_ylabel("Number of dipoles")
    fig.savefig(plot_dir / f"{base_tag}_eeg_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    col_norms = np.linalg.norm(leadfield, axis=0)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.hist(col_norms, bins=40, color="purple")
    ax.set_title("Dipole Norms Distribution")
    ax.set_xlabel("Leadfield column norm")
    ax.set_ylabel("Number of dipoles")
    fig.savefig(plot_dir / f"{base_tag}_dipole_norms.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    row_norms = np.linalg.norm(leadfield, axis=1)
    fig, ax = plt.subplots(figsize=(max(10, len(eeg_names) * 0.28), 4), constrained_layout=True)
    ax.bar(eeg_names, row_norms, color="teal")
    ax.set_title("Norm EEG Channel Sensitivity")
    ax.set_xlabel("EEG channel")
    ax.set_ylabel("Leadfield row norm")
    ax.tick_params(axis="x", labelrotation=90, labelsize=6)
    fig.savefig(plot_dir / f"{base_tag}_channel_sensitivity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def process_forward_solution(args: argparse.Namespace) -> None:
    subject = args.subject
    session = args.session
    modes = args.mode
    subject_fs = freesurfer_subject(subject)
    subjects_dir = Path(args.subjects_dir)

    os.environ["SUBJECTS_DIR"] = str(subjects_dir)

    raw_path = Path(args.raw_path) if args.raw_path else get_personalized_montage_path(subject, session)
    if not raw_path.exists():
        raise FileNotFoundError(f"Personalized montage FIF not found: {raw_path}")

    trans_path = get_trans_path(subject, session)
    fs_subject_dir = subjects_dir / subject_fs
    if not fs_subject_dir.exists():
        raise FileNotFoundError(f"FreeSurfer subject directory not found: {fs_subject_dir}")

    print(f"Processing forward solution for sub-{subject} ses-{session}")
    print(f"Modes: {', '.join(modes)}")
    print(f"Raw: {raw_path}")
    print(f"Trans: {trans_path}")
    print(f"FreeSurfer subject: {subject_fs}")

    raw = mne.io.read_raw_fif(raw_path, preload=False, verbose="ERROR")

    print("Creating BEM solution.")
    bem = create_bem_solution(subject_fs, subjects_dir)

    print("Reading coregistration transform.")
    trans = mne.read_trans(trans_path)

    for mode in modes:
        out_dir = get_forward_output_dir(subject, session, mode)
        plot_dir = out_dir / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_dir.mkdir(parents=True, exist_ok=True)
        chmod_group_dir(out_dir)
        chmod_group_dir(plot_dir)

        base_tag = get_forward_base_tag(subject, session, mode)
        fwd_path = out_dir / f"{base_tag}-fwd.fif"

        if fwd_path.exists() and not args.overwrite:
            print(f"Skipping existing forward solution: {fwd_path}")
            continue

        print(f"Creating source space for mode: {mode}")
        src = make_source_space(mode, subject_fs, subjects_dir, bem, args.spacing, args.volume_pos)
        print(src)

        print(f"Creating forward solution for mode: {mode}")
        fwd = create_forward_solution(raw.info, trans, src, bem, mindist=args.mindist)

        print(f"Writing forward solution: {fwd_path}")
        mne.write_forward_solution(fwd_path, fwd, overwrite=True)
        chmod_group(fwd_path)

        dipole_index_path = save_dipole_area_index(
            fwd,
            mode,
            subject_fs,
            subjects_dir,
            out_dir,
            base_tag,
        )
        chmod_group(dipole_index_path)

        if not args.skip_diagnostics:
            plot_forward_maps(fwd, mode, plot_dir, base_tag)
            for plot_file in plot_dir.glob(f"{base_tag}*"):
                chmod_group(plot_file)

    print(f"Finished forward solution for sub-{subject} ses-{session}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute reusable EEG forward solutions for one subject/session."
    )
    parser.add_argument("--subject", required=True, help="Subject ID without sub- prefix, e.g. 41Y01")
    parser.add_argument("--session", required=True, help="Session ID without ses- prefix, e.g. 1")
    parser.add_argument(
        "--mode",
        nargs="+",
        default=list(SOURCE_SPACE_MODES),
        choices=SOURCE_SPACE_MODES,
        help="Source space mode(s). Default: surface volume mixed.",
    )
    parser.add_argument("--spacing", default="oct6", help="Surface source-space spacing.")
    parser.add_argument("--volume-pos", type=float, default=5.0, help="Volume source-space grid spacing in mm.")
    parser.add_argument("--mindist", type=float, default=5.0, help="Minimum distance from inner skull in mm.")
    parser.add_argument("--subjects-dir", default=FREESURFER_DIR, help="FreeSurfer SUBJECTS_DIR.")
    parser.add_argument("--raw-path", default=None, help="Optional explicit personalized montage FIF.")
    parser.add_argument("--overwrite", action="store_true", help="Recompute existing forward FIFs.")
    parser.add_argument("--skip-diagnostics", action="store_true", help="Skip leadfield/sensitivity diagnostic plots.")
    return parser.parse_args()


if __name__ == "__main__":
    process_forward_solution(parse_args())
