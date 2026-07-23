#!/usr/bin/env python3
"""Create RSpre quality-control summary CSV."""

from pathlib import Path
import re

import mne
import pandas as pd


derivatives_dir = Path("/work/uphummel/studies/tTIS-EEG/derivatives/EEG")
task = "RSpre"
documentation_dir = derivatives_dir / "Documentation"
documentation_dir.mkdir(parents=True, exist_ok=True)
output_csv = documentation_dir / "preprocessing_tTIS_EEG_RSpre.csv"


def read_ica_components(path):
    """Read rejected ICA component numbers from the ICA CSV."""
    if not path.exists():
        print(f"Missing ICA CSV: {path}")
        return None

    text = path.read_text()
    components = sorted(set(int(number) for number in re.findall(r"\d+", text)))
    return components


def bad_segments_percent(raw):
    """Return percent of recording annotated as BAD_segments."""
    bad_duration = 0.0
    for description, duration in zip(raw.annotations.description, raw.annotations.duration):
        if description == "BAD_segments":
            bad_duration += float(duration)

    total_duration = raw.n_times / raw.info["sfreq"]
    return 100.0 * bad_duration / total_duration


rows = []

for task_dir in sorted(derivatives_dir.glob(f"sub-*/ses-*/{task}")):
    subject = task_dir.parts[-3].replace("sub-", "")
    session = task_dir.parts[-2].replace("ses-", "")
    base = f"sub-{subject}_ses-{session}_task-{task}_eeg"

    annotated_fif = task_dir / f"{base}_annotated_eeg.fif"
    ica_csv = task_dir / f"{base}_ica_excluded_components.csv"

    if annotated_fif.exists():
        raw = mne.io.read_raw_fif(annotated_fif, preload=False, verbose="ERROR")
        bad_segments = round(bad_segments_percent(raw), 2)
        bad_channels = raw.info["bads"]
        n_bad_channels = len(bad_channels)
        bad_channels_text = ", ".join(bad_channels)
    else:
        print(f"Missing annotated FIF: {annotated_fif}")
        bad_segments = "missing info"
        n_bad_channels = "missing info"
        bad_channels_text = "missing info"

    ica_components = read_ica_components(ica_csv)
    if ica_components is None:
        n_ica_rejected = "missing info"
        ica_rejected = "missing info"
    else:
        n_ica_rejected = len(ica_components)
        ica_rejected = ", ".join(str(component) for component in ica_components)

    rows.append(
        {
            "Patient": subject,
            "Session": session,
            "Bad segments (% omitted)": bad_segments,
            "N bad channels": n_bad_channels,
            "Bad channels": bad_channels_text,
            "N ICA rejected": n_ica_rejected,
            "Components ICA rejected": ica_rejected,
            "State": "",
            "Comments": "",
            "SS, Tony comments": "",
        }
    )


pd.DataFrame(rows).to_csv(output_csv, index=False)
print(f"Saved: {output_csv}")
