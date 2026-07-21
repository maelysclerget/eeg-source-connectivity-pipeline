"""Utility functions for 08_EEG_epoch_label_time_courses.py."""

from collections import Counter

import mne
import numpy as np
import pandas as pd


def print_annotation_descriptions(annotations):
    """Print annotation names and counts before epoching."""
    descriptions = list(annotations.description)
    if not descriptions:
        print("No annotations found in raw data.")
        return

    print("Annotations found in raw data:")
    for description, count in sorted(Counter(descriptions).items()):
        print(f"  {description}: {count}")


def labels_evoked_to_raw(evoked, annotations=None):
    """Wrap labels EvokedArray data as RawArray for MNE epoching functions."""
    raw_labels = mne.io.RawArray(evoked.data, evoked.info.copy(), verbose="ERROR") #temporarily converts labels.fif to raw obj

    if annotations is not None:
        annotations = mne.Annotations(
            onset=annotations.onset,
            duration=annotations.duration,
            description=annotations.description,
            orig_time=None,
        )
        raw_labels.set_annotations(annotations)

    return raw_labels


def _events_for_annotation(events, event_id, annotation_name):
    """Return events matching one annotation description."""
    if annotation_name not in event_id:
        raise ValueError(
            f"Could not find annotation {annotation_name!r}. "
            f"Available annotations: {sorted(event_id)}"
        )

    annotation_id = event_id[annotation_name]
    annotation_events = events[events[:, 2] == annotation_id]
    print(f"{annotation_name} events:")
    print(annotation_events)

    if len(annotation_events) == 0:
        raise ValueError(f"No events found for {annotation_name!r}.")

    return annotation_events, annotation_id


def _task_blocks_from_s15_s10(s15_events, s10_events):
    """Return task blocks defined by each S15 and the first following S10."""
    blocks = []
    for block_index, s15_event in enumerate(s15_events, start=1):
        s15_sample = int(s15_event[0])
        s10_sample = int(s10_events[block_index - 1, 0])

        blocks.append(
            {
                "block": block_index,
                "s15": s15_sample,
                "s10": s10_sample,
            }
        )

    return blocks


def _fixed_epochs_from_block_starts(raw_labels, block_starts, condition, epoch_duration, epochs_per_block, event_base): # we don't have the code but know the start sample: Task non-stim baseline, Task stim baseline, Task fixed block, RSpre / RSpost
    """Create fixed-length epochs and keep block labels in metadata."""
    sfreq = raw_labels.info["sfreq"]
    epoch_samples = int(round(epoch_duration * sfreq)) #converts epoch duration from seconds into samples
    events = []
    metadata = []
    event_id = {}

    for block, start_sample in block_starts:
        event_name = f"{condition}/block_{block:02d}"
        event_code = event_base + block #creates a numeric event code for MNE
        event_id[event_name] = event_code
        for epoch_in_block in range(1, epochs_per_block + 1): 
            sample = int(start_sample + (epoch_in_block - 1) * epoch_samples) # epoch 1 starts at start_sample, epoch 2 starts 5 seconds later, epoch 3 starts 10 seconds later
            events.append([sample, 0, event_code]) #[sample, previous_value, event_code]
            metadata.append(
                {
                    "condition": condition,
                    "block": block,
                    "epoch_in_block": epoch_in_block,
                    "duration_s": epoch_duration,
                }
            )

    return mne.Epochs(
        raw_labels,
        events=np.asarray(events, dtype=int),
        event_id=event_id,
        tmin=0.0,
        tmax=epoch_duration - 1.0 / sfreq,
        baseline=None,
        metadata=pd.DataFrame(metadata),
        preload=True,
        reject=None,
        flat=None,
        reject_by_annotation=False,
        verbose="ERROR",
    )


def _fixed_epochs_from_events(raw_labels, epoch_specs, condition, epoch_duration, event_base): #for RS: we have the code we build a list of epochs and use this list to create epochs
    """Create fixed-length epochs from explicit starts with metadata."""
    sfreq = raw_labels.info["sfreq"]
    events = []
    metadata = []
    event_id = {}

    for block, epoch_in_block, start_sample in epoch_specs:
        event_name = f"{condition}/block_{block:02d}"
        event_code = event_base + block
        event_id[event_name] = event_code
        events.append([int(start_sample), 0, event_code])
        metadata.append(
            {
                "condition": condition,
                "block": block,
                "epoch_in_block": epoch_in_block,
                "duration_s": epoch_duration,
            }
        )

    return mne.Epochs(
        raw_labels,
        events=np.asarray(events, dtype=int),
        event_id=event_id,
        tmin=0.0,
        tmax=epoch_duration - 1.0 / sfreq,
        baseline=None,
        metadata=pd.DataFrame(metadata),
        preload=True,
        reject=None,
        flat=None,
        reject_by_annotation=False,
        verbose="ERROR",
    )


def make_task_epochs(
    raw_labels,
    epoch_duration=5.0,
    s15_annotation="Stimulus/S 15",
    block_start_annotation="Stimulus/S 10",
):
    """Create requested 5 s task baseline, fixed-block, and sequence epochs."""
    events, event_id = mne.events_from_annotations(raw_labels, verbose="ERROR") 
    print("Event IDs found:", event_id)

    s15_events, _ = _events_for_annotation(events, event_id, s15_annotation)
    s10_events, _ = _events_for_annotation(events, event_id, block_start_annotation)

    sfreq = raw_labels.info["sfreq"]
    task_blocks = _task_blocks_from_s15_s10(s15_events, s10_events)

    no_stim_starts = [(block["block"], block["s15"] - int(round(20.0 * sfreq))) for block in task_blocks]
    stim_starts = [(block["block"], block["s15"]) for block in task_blocks]
    fixed_task_starts = [(block["block"], block["s10"]) for block in task_blocks]

    baseline_no_stim_epochs = _fixed_epochs_from_block_starts(
        raw_labels, no_stim_starts, "baseline_no_stim", epoch_duration, 4, 100
    )
    baseline_stim_epochs = _fixed_epochs_from_block_starts(
        raw_labels, stim_starts, "baseline_stim", epoch_duration, 5, 200
    )
    fixed_task_epochs = _fixed_epochs_from_block_starts(
        raw_labels, fixed_task_starts, "task_fixed", epoch_duration, 18, 300
    )
    sequence_epochs = make_task_sequence_epochs(
        raw_labels, task_blocks, s10_events, epoch_duration, task_duration=90.0
    )

    return baseline_no_stim_epochs, baseline_stim_epochs, fixed_task_epochs, sequence_epochs


def make_rs_epochs(raw_labels, task_name, epoch_duration=5.0, s15_annotation="Stimulus/S 15"):
    """Create 5 s resting-state epochs after S15 and keep S15 block labels."""
    events, event_id = mne.events_from_annotations(raw_labels, verbose="ERROR")
    print("Event IDs found:", event_id)
    s15_events, _ = _events_for_annotation(events, event_id, s15_annotation)

    sfreq = raw_labels.info["sfreq"]
    recording_end_sample = raw_labels.n_times - 1
    epoch_specs = []
    epoch_samples = int(round(epoch_duration * sfreq))

    for block_index, s15_event in enumerate(s15_events, start=1):
        start_sample = int(s15_event[0])
        next_s15_sample = int(s15_events[block_index, 0]) if block_index < len(s15_events) else recording_end_sample
        n_epochs = max(0, (next_s15_sample - start_sample) // epoch_samples)
        for epoch_in_block in range(1, n_epochs + 1):
            epoch_specs.append((block_index, epoch_in_block, start_sample + (epoch_in_block - 1) * epoch_samples))

    return _fixed_epochs_from_events(
        raw_labels,
        epoch_specs,
        f"{task_name}_s15",
        epoch_duration,
        500,
    )


def make_rs_baseline_epochs(raw_labels, epoch_duration=5.0, baseline_duration=20.0):
    """Create RSpre/RSpost baseline from the first 20 s as 5 s epochs."""
    n_epochs = int(round(baseline_duration / epoch_duration))
    return _fixed_epochs_from_block_starts(
        raw_labels,
        [(1, 0)],
        "rs_baseline_first20",
        epoch_duration,
        n_epochs,
        600,
    )


def make_task_sequence_epochs(raw_labels, task_blocks, s10_events, epoch_duration=5.0, task_duration=90.0):
    """Create one variable-length Epochs object per S10-to-next-S10 sequence interval."""
    sfreq = raw_labels.info["sfreq"]
    task_samples = int(round(task_duration * sfreq))
    sequence_epochs = []

    for block in task_blocks:
        block_number = block["block"]
        task_start = block["s10"]
        task_stop = task_start + task_samples
        block_s10 = [int(event[0]) for event in s10_events if task_start <= int(event[0]) < task_stop]

        for sequence_index, start_sample in enumerate(block_s10, start=1):
            if sequence_index < len(block_s10):
                stop_sample = block_s10[sequence_index]
            else:
                stop_sample = task_stop
            duration = (stop_sample - start_sample) / sfreq
            event_id = {f"task_sequence/block_{block_number:02d}/seq_{sequence_index:02d}": 700 + block_number * 100 + sequence_index}
            metadata = pd.DataFrame(
                [
                    {
                        "condition": "task_sequence",
                        "block": block_number,
                        "sequence": sequence_index,
                        "duration_s": duration,
                    }
                ]
            )
            epochs = mne.Epochs(
                raw_labels,
                events=np.array([[start_sample, 0, next(iter(event_id.values()))]], dtype=int),
                event_id=event_id,
                tmin=0.0,
                tmax=duration - 1.0 / sfreq,
                baseline=None,
                metadata=metadata,
                preload=True,
                reject=None,
                flat=None,
                reject_by_annotation=False,
                verbose="ERROR",
            )
            sequence_epochs.append((block_number, sequence_index, duration, epochs))

    return sequence_epochs
def save_epochs(epochs, path):
    """Save Epochs FIF."""
    epochs.save(path, overwrite=True)
    print(f"Saved {len(epochs)} epochs: {path}")
