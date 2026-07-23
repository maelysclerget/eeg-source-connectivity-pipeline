#!/usr/bin/env python3
from pathlib import Path


derivatives_dir = Path("/work/uphummel/studies/tTIS-EEG/derivatives/EEG")

old_endings = [
    "_eeg_final_preprocessed_eeg_not_interpolated.fif",
    "_eeg_final_preprocessed_raw_source_level.fif",
]

new_ending = "_eeg_not_interpolated_final_preprocessed_eeg.fif"

old_interpolated_endings = [
    "_eeg_final_preprocessed_raw_sensor_level.fif",
    "_eeg_final_preprocessed_eeg_interpolated.fif",
]

new_interpolated_ending = "_eeg_interpolated_final_preprocessed_eeg.fif"


for subject_dir in sorted(derivatives_dir.glob("sub-*")):
    for session_dir in sorted(subject_dir.glob("ses-*")):
        for task_dir in sorted(session_dir.iterdir()):
            if not task_dir.is_dir() or task_dir.name == "source_reconstruction":
                continue

            subject = subject_dir.name.replace("sub-", "")
            session = session_dir.name.replace("ses-", "")
            task = task_dir.name
            stem = f"sub-{subject}_ses-{session}_task-{task}"

            for old_ending in old_endings:
                old_file = task_dir / f"{stem}{old_ending}"
                new_file = task_dir / f"{stem}{new_ending}"

                if old_file.exists() and not new_file.exists():
                    old_file.rename(new_file)
                    print(f"{old_file} -> {new_file}")

            for old_ending in old_interpolated_endings:
                old_file = task_dir / f"{stem}{old_ending}"
                new_file = task_dir / f"{stem}{new_interpolated_ending}"

                if old_file.exists() and not new_file.exists():
                    old_file.rename(new_file)
                    print(f"{old_file} -> {new_file}")
