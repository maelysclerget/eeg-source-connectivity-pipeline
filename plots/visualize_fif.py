#!/usr/bin/env python3
# Run this on 'on demand'
"""
Visualize MNE inverse-solution FIF outputs.

Run with:
    python visualize_fif.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import mne


subject = "41Y01"
session = "1"
mode = "surface"
method = "MNE"
cov = "cov-s15"

subject_fs = f"sub-{subject}_ses-baseline"

source_reconstruction_dir = Path(f"/work/uphummel/studies/tTIS-EEG/derivatives/EEG/sub-{subject}/ses-{session}/source_reconstruction")

inverse_dir = (
    source_reconstruction_dir
    / "inverse_solution"
    / mode
    / method
    / cov
)

trans_path = source_reconstruction_dir / f"{subject}_ses{session}_trans.fif"

subjects_dir = Path("/work/uphummel/studies/tTIS-EEG/derivatives/MRI/freesurfer")


def find_one(folder: Path, suffix: str):
    """Return the first file in folder matching suffix, or None if absent."""
    files = sorted(folder.glob(f"*{suffix}"))

    if len(files) == 0:
        print(f"No {suffix} file found.")
        return None

    if len(files) > 1:
        print(f"Several {suffix} files found. Using: {files[0].name}")

    return files[0]


def show_forward(fwd_path: Path):
    """Load a forward solution, print its metadata, and plot its source space."""    
    print("\nOpening forward solution:")
    print(fwd_path)

    fwd = mne.read_forward_solution(fwd_path)
    print(fwd)
    print("Leadfield shape:", fwd["sol"]["data"].shape)

    fwd["src"].plot(
        subjects_dir=subjects_dir,
        trans=trans_path,
    )


def show_cov(cov_path: Path, fwd_path: Path | None = None):
    """Load a covariance file, print its metadata, and plot it with optional forward info."""
    print("\nOpening covariance:")
    print(cov_path)

    cov = mne.read_cov(cov_path)
    print(cov)

    if fwd_path is not None:
        fwd = mne.read_forward_solution(fwd_path)
        info = fwd["info"]
        cov.plot(info, proj=False, show=True)
    else:
        cov.plot(show_svd=False)
        

def show_inverse(inv_path: Path):
    """Load an inverse operator, print its metadata, and plot its source space."""
    print("\nOpening inverse operator:")
    print(inv_path)

    inv = mne.minimum_norm.read_inverse_operator(inv_path)
    print(inv)

    inv["src"].plot(subjects_dir=subjects_dir, trans=trans_path)


def show_source_estimate():
    """Load and plot the method/mode/cov surface source estimate pair."""
    stc_prefix = (
        inverse_dir
        / f"{subject}_ses{session}_task-task_src-{mode}_method-{method}_{cov}_stc"
    )
    stc_lh_path = Path(f"{stc_prefix}-lh.stc")
    stc_rh_path = Path(f"{stc_prefix}-rh.stc")

    if not stc_lh_path.exists() or not stc_rh_path.exists():
        print("\nNo matching STC pair found:")
        print(stc_lh_path)
        print(stc_rh_path)
        return

    print("\nOpening source estimate:")
    print(stc_prefix)

    stc = mne.read_source_estimate(stc_prefix)
    print(stc)

    stc.plot(
        subject=subject_fs,
        subjects_dir=subjects_dir,
    )


def show_labels():
    """Load and plot the method/mode/cov labels FIF file."""
    labels_path = (
        inverse_dir
        / f"{subject}_ses{session}_task-task_src-{mode}_method-{method}_{cov}_labels.fif"
    )

    if not labels_path.exists():
        print("\nNo matching labels FIF found:")
        print(labels_path)
        return

    print("\nInspecting labels file:")
    print(labels_path)

    file_type = mne.what(labels_path)
    print("MNE file type:", file_type)

    if file_type == "evoked":
        evokeds = mne.read_evokeds(labels_path)
        for evoked in evokeds:
            print(evoked)
            
            roi_names = evoked.ch_names
            sfreq = evoked.info["sfreq"]
            roi_data = evoked.data

            info = mne.create_info(ch_names=roi_names, sfreq=sfreq, ch_types="misc")

            evoked_array = mne.EvokedArray(roi_data, info)

            print("\nReconstructed ROI Info:")
            print(info)
            print("\nReconstructed ROI EvokedArray:")
            print(evoked_array)

            evoked.plot(spatial_colors=False, show=False)
        return
    mne.io.show_fiff(labels_path)


def main():
    print("Inverse directory:")
    print(inverse_dir)

    print("\nTransform file:")
    print(trans_path)

    if not inverse_dir.exists():
        raise FileNotFoundError(f"Directory not found: {inverse_dir}")

    if not trans_path.exists():
        raise FileNotFoundError(f"Transform file not found: {trans_path}")

    fwd_path = find_one(inverse_dir, "-fwd.fif")
    #cov_path = find_one(inverse_dir, "-cov.fif")
    #inv_path = find_one(inverse_dir, "-inv.fif")
    #show_source_estimate()
    show_labels()

    if fwd_path is not None:
        show_forward(fwd_path)

    #if cov_path is not None:
    #    show_cov(cov_path, fwd_path=fwd_path)

    #if inv_path is not None:
    #    show_inverse(inv_path)

    print("\nClose the figures/windows to exit.")
    plt.show()


if __name__ == "__main__":
    main()
