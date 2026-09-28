#!/usr/bin/env python3
# Author Stavriani Skarvelaki / Maelys Clerget
"""
Script for EEG source localization STEP 2 (after coregistration):

1. Load the personalized EEG montage created by 05_EEG_montage_BEM.py.
2. Load the reusable forward solution created by 06_EEG_forward_solution.py.
3. Create the inverse operator, compute source estimates, and save ROI time courses.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from collections import Counter

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne
import numpy as np

from utils06 import (
    create_inverse_operator,
    apply_source_morph,
    compute_surface_source_morph,
    extract_labels_all_modes,
    save_label_ts_as_fif,
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


def get_forward_path(subject: str, session: str, mode: str) -> Path:
    return (
        Path(DERIVATIVES_DIR)
        / f"sub-{subject}"
        / f"ses-{session}"
        / "source_reconstruction"
        / "forward_solution"
        / mode
        / f"{subject}_ses{session}_src-{mode}-fwd.fif"
    )


def plot_noise_covariance(noise_cov: mne.Covariance, info: mne.Info, plot_dir: Path, base_tag: str) -> None:
    """Save EEG covariance matrix and eigenvalue diagnostic plots."""
    plot_dir.mkdir(parents=True, exist_ok=True)
    print("Saving EEG noise covariance matrix plots.")

    cov_data = np.asarray(noise_cov.data)
    if cov_data.ndim == 1:
        cov_data = np.diag(cov_data)

    n_channels = cov_data.shape[0]
    fig_size = max(8, n_channels * 0.18)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size), constrained_layout=True)
    im = ax.imshow(cov_data, cmap="RdBu_r", aspect="equal")
    ax.set_title("EEG covariance")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(
        plot_dir / f"{base_tag}_covariance.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    eigvals = np.linalg.eigvalsh(cov_data)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(np.arange(1, len(eigvals) + 1), np.sort(eigvals)[::-1], marker=".")
    ax.set_title("EEG covariance eigenvalues")
    ax.set_xlabel("Eigenvalue rank")
    ax.set_ylabel("Eigenvalue")
    fig.savefig(
        plot_dir / f"{base_tag}_eigenvalues.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

# Takes the source reconstruction result and saves it as ROI-level time courses.
def save_roi_time_courses(stc, fwd, mode: str, subject_fs: str, subjects_dir: Path, out_dir: Path, base_tag: str) -> None:
    """Extract ROI-level source time courses and save them as FIF files."""
    print("Extracting label time series.")
    src_fwd = fwd["src"]

    if mode == "surface":
        labels, label_ts, _ = extract_labels_all_modes(
            stc, src_fwd, mode, subject_fs, subjects_dir
        )
        roi_names = [label.name for label in labels]
        save_label_ts_as_fif(label_ts, roi_names, stc, out_dir / f"{base_tag}_labels.fif")
    elif mode == "volume":
        roi_names, roi_ts, _ = extract_labels_all_modes(
            stc, src_fwd, mode, subject_fs, subjects_dir
        )
        save_label_ts_as_fif(roi_ts, roi_names, stc, out_dir / f"{base_tag}_labels.fif")
    elif mode == "mixed":
        roi_dict, _, _ = extract_labels_all_modes(
            stc, src_fwd, mode, subject_fs, subjects_dir
        )
        save_label_ts_as_fif(
            roi_dict["surface_ts"],
            [label.name for label in roi_dict["surface_labels"]],
            stc,
            out_dir / f"{base_tag}_surface_labels.fif",
        )
        save_label_ts_as_fif(
            roi_dict["volume_ts"],
            roi_dict["volume_labels"],
            stc,
            out_dir / f"{base_tag}_volume_labels.fif",
        )

def print_annotation_descriptions(raw: mne.io.BaseRaw) -> None:
    """Print annotation names so task-specific covariance windows can be checked."""
    descriptions = list(raw.annotations.description)
    if not descriptions:
        print("No annotations found in raw data.")
        return

    print("Annotations found in raw data:")
    for description, count in sorted(Counter(descriptions).items()):
        print(f"  {description}: {count}")


def annotation_matches(description: str, marker: str) -> bool:
    """Match annotation names after lowercasing and removing spaces."""
    normalized_description = description.lower().replace(" ", "")
    normalized_marker = marker.lower().replace(" ", "")
    return normalized_description == normalized_marker


def annotation_onsets(raw: mne.io.BaseRaw, marker: str) -> list[float]:
    """Return all annotation onsets matching marker."""
    matches = []
    for description, onset in zip(raw.annotations.description, raw.annotations.onset):
        if annotation_matches(description, marker):
            matches.append(float(onset))

    if matches:
        print(
            f"Using {len(matches)} annotation(s) matching {marker} for covariance: "
            f"{matches[0]:.3f}-{matches[-1]:.3f} s."
        )
        return matches

    available = sorted(set(raw.annotations.description))
    raise ValueError(
        f"Could not find annotation matching {marker!r}. "
        f"Available annotations: {available}"
    )


def crop_raw_for_covariance(raw: mne.io.BaseRaw, crop_start: float, crop_duration: float) -> mne.io.BaseRaw:
    """Return a copy of raw cropped to the covariance time window."""
    tmin = crop_start
    tmax = min(crop_start + crop_duration, raw.times[-1])
    if tmin >= raw.times[-1]:
        raise ValueError(
            f"covariance start={crop_start} is outside the recording, "
            f"which ends at {raw.times[-1]:.3f} s."
        )

    print(f"Using raw data from {tmin:.3f}-{tmax:.3f} s for covariance.")
    return raw.copy().crop(tmin=tmin, tmax=tmax)

def concatenate_covariance_windows(raw: mne.io.BaseRaw, crop_starts: list[float], crop_duration: float,) -> mne.io.BaseRaw:
    """Return a raw object made from all covariance windows concatenated together."""
    windows = [crop_raw_for_covariance(raw, crop_start, crop_duration) for crop_start in crop_starts]
    if len(windows) == 1:
        return windows[0]

    print(
        f"Concatenating {len(windows)} covariance windows "
        f"({crop_duration:.3f} s each before end-of-recording clipping)."
    )
    return mne.concatenate_raws(windows)


def make_noise_covariance_windows(raw: mne.io.BaseRaw, task: str, duration: float):
    """Define task-specific covariance windows as label, starts, duration tuples."""
    task_lower = task.lower()
    if task_lower == "task":
        return [
            ("s4", annotation_onsets(raw, "Stimulus/S  4"), duration),
            ("s15", annotation_onsets(raw, "Stimulus/S 15"), duration),
        ]
    return [("first20", [0.0], duration)]

def get_inverse_output_dir(subject: str, session: str, task: str, mode: str, method: str, cov_label: str) -> Path:
    """Return the output directory for one task/covariance combination."""
    base_dir = (
        Path(DERIVATIVES_DIR)
        / f"sub-{subject}"
        / f"ses-{session}"
        / task
        / "inverse_solution"
        / mode
        / method
    )

    task_lower = task.lower()
    if task_lower == "task":
        return base_dir / f"cov{cov_label}"
    return base_dir


def get_inverse_base_tag(subject: str, session: str, task: str, mode: str, method: str, cov_label: str) -> str:
    """Return the filename stem for one inverse-solution output set."""
    base_tag = f"{subject}_ses{session}_task-{task}_src-{mode}_method-{method}"
    if task.lower() != "task":
        return base_tag
    return f"{base_tag}_cov-{cov_label}"

def compute_noise_covariances(raw: mne.io.BaseRaw, task: str, duration: float) -> dict[str, mne.Covariance]:
    """Compute all noise covariance matrices needed for the requested task."""
    covariances = {}
    for cov_label, cov_starts, cov_duration in make_noise_covariance_windows(raw, task, duration):
        print(f"Computing noise covariance matrix: {cov_label}")
        cov_raw = concatenate_covariance_windows(raw, cov_starts, cov_duration)
        covariances[cov_label] = mne.compute_raw_covariance(cov_raw)
    return covariances


def compute_source_time_courses(raw, fwd, noise_cov, mode: str, method: str, snr: float):
    """Create and apply the selected minimum-norm inverse operator."""
    if method not in {"MNE", "sLORETA", "eLORETA"}:
        raise ValueError("method must be 'MNE', 'sLORETA', or 'eLORETA'")

    print(f"Creating {method} inverse operator.")
    inverse_operator = create_inverse_operator(raw.info, fwd, noise_cov, mode)

    lambda2 = 1.0 / (snr**2)
    print(f"Computing source time courses using {method}, SNR={snr}, lambda2={lambda2}.")
    stc = mne.minimum_norm.apply_inverse_raw(
        raw,
        inverse_operator,
        lambda2=lambda2,
        method=method,
    )
    return stc, inverse_operator

def save_full_stc(stc, out_dir: Path, base_tag: str, mode: str) -> None:
    """Save the full source estimate using a format compatible with the source space."""
    if mode == "surface":
        stc_path = out_dir / f"{base_tag}_stc"
        stc.save(stc_path, overwrite=True)
    else:
        stc_path = out_dir / f"{base_tag}_stc.h5"
        stc.save(stc_path, ftype="h5", overwrite=True)

    for saved_file in out_dir.glob(f"{base_tag}_stc*"):
        chmod_group(saved_file)
        
    print(f"Saved STC to {stc_path}")


def process_inverse_solution(args: argparse.Namespace) -> None:
    subject = args.subject
    session = args.session
    task = args.task
    modes = args.mode
    subject_fs = freesurfer_subject(subject)
    subjects_dir = Path(args.subjects_dir)

    os.environ["SUBJECTS_DIR"] = str(subjects_dir)

    raw_path = Path(args.raw_path) if args.raw_path else get_personalized_montage_path(subject, session)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw FIF not found: {raw_path}")

    trans_path = get_trans_path(subject, session)
    fs_subject_dir = subjects_dir / subject_fs
    if not fs_subject_dir.exists():
        raise FileNotFoundError(f"FreeSurfer subject directory not found: {fs_subject_dir}")

    print(f"Processing sub-{subject} ses-{session} task-{task}")
    print(f"Modes: {', '.join(modes)}")
    print(f"Raw: {raw_path}")
    print(f"Trans: {trans_path}")
    print(f"FreeSurfer subject: {subject_fs}")

    raw = mne.io.read_raw_fif(raw_path, preload=True, verbose="ERROR")
    print_annotation_descriptions(raw)
    noise_covariances = compute_noise_covariances(raw, task, args.cov_duration)

    print("Reading coregistration transform.")
    mne.read_trans(trans_path)

    for mode in modes:
        fwd_path = get_forward_path(subject, session, mode)
        if not fwd_path.exists():
            raise FileNotFoundError(
                f"Forward solution not found: {fwd_path}. "
                "Run 06_EEG_forward_solution.py before the inverse step."
            )

        print(f"Reading reusable forward solution for mode={mode}: {fwd_path}")
        fwd = mne.read_forward_solution(fwd_path, verbose="ERROR")

        for cov_label, noise_cov in noise_covariances.items():
            out_dir = get_inverse_output_dir(subject, session, task, mode, args.method, cov_label)
            plot_dir = out_dir / "plots"
            out_dir.mkdir(parents=True, exist_ok=True)
            plot_dir.mkdir(parents=True, exist_ok=True)
            os.chmod(out_dir, 0o2770)
            os.chmod(plot_dir, 0o2770)

            base_tag = get_inverse_base_tag(subject, session, task, mode, args.method, cov_label)

            print(f"Output: {out_dir}")

            cov_path = out_dir / f"{base_tag}-cov.fif"
            mne.write_cov(cov_path, noise_cov, overwrite=True)
            chmod_group(cov_path)

            if not args.skip_diagnostics:
                plot_noise_covariance(noise_cov, raw.info, plot_dir, base_tag)

            stc, operator = compute_source_time_courses(
                raw,
                fwd,
                noise_cov,
                mode,
                args.method,
                args.snr,
            )

            inv_path = out_dir / f"{base_tag}-inv.fif"
            mne.minimum_norm.write_inverse_operator(inv_path, operator, overwrite=True)
            chmod_group(inv_path)

            try:
                save_full_stc(stc, out_dir, base_tag, mode)
            except Exception as err:
                print(f"Could not save full STC ({err}). Continuing with ROI time courses.")

            if args.morph_to_fsaverage:
                if mode != "surface":
                    print(
                        f"Skipping source morph for mode={mode}. "
                        "This implementation morphs cortical surface STCs only."
                    )
                else:
                    print(
                        f"Computing source morph: {subject_fs} -> fsaverage (ico5)."
                    )
                    morph, src_to = compute_surface_source_morph(
                        stc,
                        subject_from=subject_fs,
                        subjects_dir=subjects_dir,
                    )
                    stc_morph = apply_source_morph(stc, morph)

                    morph_tag = f"{base_tag}_morph-fsaverage"

                    if src_to is not None:
                        src_to_path = out_dir / f"{morph_tag}-src.fif"
                        mne.write_source_spaces(src_to_path, src_to, overwrite=True)
                        chmod_group(src_to_path)

                    stc_morph_path = out_dir / f"{morph_tag}_stc"
                    stc_morph.save(stc_morph_path, overwrite=True)
                    print(f"Saved morphed STC to {stc_morph_path}")

            save_roi_time_courses(stc, fwd, mode, subject_fs, subjects_dir, out_dir, base_tag)

            for output_file in out_dir.glob(f"{base_tag}*"):
                chmod_group(output_file)

    print(f"Finished inverse solution for sub-{subject} ses-{session} task-{task}.")
 


#function that reads the options you pass from the command line.
def parse_args() -> argparse.Namespace:
    """Parse command-line options for one inverse-solution job."""
    parser = argparse.ArgumentParser(
        description="Compute EEG forward/inverse solution and ROI source time courses."
    )
    parser.add_argument("--subject", required=True, help="Subject ID without sub- prefix, e.g. 41Y01")
    parser.add_argument("--session", required=True, help="Session ID without ses- prefix, e.g. 1")
    parser.add_argument("--task", required=True, help="Task name without task- prefix, e.g. task")
    parser.add_argument(
        "--mode",
        nargs="+",
        default=list(SOURCE_SPACE_MODES),
        choices=SOURCE_SPACE_MODES,
        help="Source space mode(s). Default: surface volume mixed.",
    )
    parser.add_argument(
        "--method",
        default="sLORETA",
        choices=["MNE", "sLORETA", "eLORETA"],
        help="Minimum-norm inverse method.",
    )
    parser.add_argument("--snr", type=float, default=3.0, help="SNR used to compute lambda2=1/SNR^2.")
    parser.add_argument(
        "--cov-duration",
        type=float,
        default=20.0,
        help="Duration in seconds used to compute each noise covariance.",
    )
    parser.add_argument("--spacing", default="oct6", help="Surface source-space spacing.")
    parser.add_argument("--volume-pos", type=float, default=5.0, help="Volume source-space grid spacing in mm.")
    parser.add_argument("--mindist", type=float, default=5.0, help="Minimum distance from inner skull in mm.")
    parser.add_argument("--subjects-dir", default=FREESURFER_DIR, help="FreeSurfer SUBJECTS_DIR.")
    parser.add_argument("--raw-path", default=None, help="Optional explicit personalized montage FIF.")
    parser.add_argument("--morph-to-fsaverage", action="store_true", help="Morph cortical surface STCs to fsaverage using an ico5 target source space.",)
    parser.add_argument("--skip-diagnostics", action="store_true", help="Skip leadfield/sensitivity diagnostic plots.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    process_inverse_solution(parse_args())
