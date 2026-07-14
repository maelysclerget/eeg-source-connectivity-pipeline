#!/usr/bin/env python3
# Author Stavriani Skarvelaki / Maelys Clerget

"""Save a static plot of the volume source estimate."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne


subject = "41Y01"
session = "1"
task = "RSpre"
mode = "volume"
method = "eLORETA"
subject_fs = f"sub-{subject}_ses-baseline"

subjects_dir = Path("/work/uphummel/studies/tTIS-EEG/derivatives/MRI/freesurfer")
inverse_dir = (
    Path("/work/uphummel/studies/tTIS-EEG/derivatives/EEG")
    / f"sub-{subject}"
    / f"ses-{session}"
    / "source_reconstruction"
    / "inverse_solution"
    / mode
    / method
    / task
)

base_tag = f"{subject}_ses{session}_task-{task}_src-{mode}_method-{method}"
stc_path = inverse_dir / f"{base_tag}_stc.h5"
fwd_path = inverse_dir / f"{base_tag}-fwd.fif"

output_dir = inverse_dir / "plots"
output_dir.mkdir(exist_ok=True)

initial_time = 0.09

print(f"Reading volume STC: {stc_path}")
stc = mne.read_source_estimate(stc_path, subject=subject_fs)
print(stc)
print(f"Loaded STC data shape: {stc.data.shape}")

print(f"Keeping one time point for plotting: {initial_time:.3f}s")
stc = stc.copy().crop(tmin=initial_time, tmax=initial_time)
print(f"Cropped STC data shape: {stc.data.shape}")

print(f"Reading forward solution for volume source space: {fwd_path}")
fwd = mne.read_forward_solution(fwd_path)
src = fwd["src"]

print("Plotting volume STC...")
fig = stc.plot(
    src,
    subject=subject_fs,
    subjects_dir=subjects_dir,
    initial_time=initial_time,
    clim=dict(kind="percent", lims=[70, 85, 99]),
    mode="stat_map",
    verbose="error",
)

output_path = output_dir / f"{base_tag}_volume_stc_t0090.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {output_path}")
