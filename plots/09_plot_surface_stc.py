#!/usr/bin/env python3
# Author Stavriani Skarvelaki / Maelys Clerget
# Run it with sbatch 09_run_plot_surface_stc.sh

"""Save a static plot of the full morphed fsaverage STC."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne


subject = "41Y01"
session = "1"
task = "RSpre"
mode = "surface"
method = "eLORETA"

subjects_dir = Path("/work/uphummel/studies/tTIS-EEG/derivatives/MRI/freesurfer")
script_dir = Path(__file__).resolve().parent

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
stc_path = inverse_dir / f"{base_tag}_morph-fsaverage_stc"

output_dir = script_dir / "surface_stc_plots"
output_dir.mkdir(parents=True, exist_ok=True)

initial_time = 1.0

print("Reading full morphed STC...")
stc_fsaverage = mne.read_source_estimate(stc_path, subject="fsaverage")
print(f"Loaded STC data shape: {stc_fsaverage.data.shape}")

print("Plotting with Matplotlib backend...")
fig = stc_fsaverage.plot(
    subjects_dir=subjects_dir,
    initial_time=initial_time,
    backend="matplotlib",
    verbose="error",
    smoothing_steps=7,
    hemi="lh",
    views="lat", 
    clim=dict(kind="percent", lims=[70, 85, 99])
)

output_path = output_dir / f"{base_tag}_surface_stc_t1000.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {output_path}")
