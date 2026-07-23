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


def renamed_file(old_file, old_ending, new_ending, task_dir):
    """Build the new filename, including the special task_v1 folder name."""
    new_name = old_file.name.replace(old_ending, new_ending)
    if task_dir.name == "task_v1":
        new_name = new_name.replace("_task-task_", "_task-task_v1_")
    return old_file.with_name(new_name)


for subject_dir in sorted(derivatives_dir.glob("sub-*")):
    for session_dir in sorted(subject_dir.glob("ses-*")):
        for task_dir in sorted(session_dir.iterdir()):
            if not task_dir.is_dir() or task_dir.name == "source_reconstruction":
                continue

            for old_ending in old_endings:
                for old_file in sorted(task_dir.glob(f"*{old_ending}")):
                    new_file = renamed_file(old_file, old_ending, new_ending, task_dir)

                    if not new_file.exists():
                        old_file.rename(new_file)
                        print(f"{old_file} -> {new_file}")

            for old_ending in old_interpolated_endings:
                for old_file in sorted(task_dir.glob(f"*{old_ending}")):
                    new_file = renamed_file(old_file, old_ending, new_interpolated_ending, task_dir)

                    if not new_file.exists():
                        old_file.rename(new_file)
                        print(f"{old_file} -> {new_file}")
