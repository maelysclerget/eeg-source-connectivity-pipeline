#!/usr/bin/env python3
"""Print ROI/channel counts for saved labels FIF files."""

from pathlib import Path

import mne


derivatives_dir = Path("/work/uphummel/studies/tTIS-EEG/derivatives/EEG")
subject = "41Y01"
session = "1"
task = "RSpre"
mode = "mixed"
method = "MNE"


inverse_dir = (
    derivatives_dir
    / f"sub-{subject}"
    / f"ses-{session}"
    / "source_reconstruction"
    / "inverse_solution"
    / mode
    / method
    / task
)


print(f"Checking labels FIF files in: {inverse_dir}")

for labels_path in sorted(inverse_dir.glob("*labels.fif")):
    evokeds = mne.read_evokeds(labels_path, verbose="ERROR")
    evoked = evokeds[0]

    n_roi_ts = evoked.data.shape[0]
    n_names = len(evoked.ch_names)
    n_times = evoked.data.shape[1]

    print()
    print(labels_path.name)
    print(f"  n_roi_ts = {n_roi_ts}")
    print(f"  n_names  = {n_names}")
    print(f"  n_times  = {n_times}")
    print(f"  first names: {evoked.ch_names[:10]}")

    if n_roi_ts != n_names:
        print("  MISMATCH")
    else:
        print("  OK")
