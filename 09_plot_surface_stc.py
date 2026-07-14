#!/usr/bin/env python3
# Author Stavriani Skarvelaki / Maelys Clerget

"""Save a static plot of the full morphed fsaverage STC."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mne


subjects_dir = Path("/work/uphummel/studies/tTIS-EEG/derivatives/MRI/freesurfer")

stc_path = (
    "/work/uphummel/studies/tTIS-EEG/derivatives/EEG/sub-41Y01/ses-1/"
    "source_reconstruction/inverse_solution/surface/eLORETA/RSpre/"
    "41Y01_ses1_task-RSpre_src-surface_method-eLORETA_morph-fsaverage_stc"
)

output_dir = Path(stc_path).parent / "plots"
output_dir.mkdir(exist_ok=True)

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

output_path = output_dir / "morphed_fsaverage_matplotlib_full.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {output_path}")

