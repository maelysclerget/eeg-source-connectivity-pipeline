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


def labels_evoked_to_raw(evoked, annotations):
    """Wrap labels EvokedArray data as RawArray for MNE epoching functions."""
    # evoked.data is the continuous source/ROI labels time course.
    # Convert it to RawArray because MNE epochs Raw objects, not Evoked objects.
    raw_labels = mne.io.RawArray(evoked.data, evoked.info.copy(), verbose="ERROR") #temporarily converts labels.fif to raw obj

    # These annotations come from the preprocessed raw EEG file.
    # Attach them to the source/ROI time course so MNE knows where to epoch it.
    annotations = mne.Annotations(
        onset=annotations.onset,
        duration=annotations.duration,
        description=annotations.description,
        orig_time=None,
    )
    # This is the line that puts the raw EEG annotations onto labels.fif data.
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
        s15_sample = int(s15_event[0]) # S15 sample
        # block_index starts at 1, but Python indices start at 0.
        # Example: block 1 uses s10_events[0].
        s10_sample = int(s10_events[block_index - 1, 0]) # S10 sample

        blocks.append(
            {
                "block": block_index,
                "s15": s15_sample,
                "s10": s10_sample,
            }
        )

    return blocks


def _make_task_block_epochs(raw_labels, block_starts, condition, epoch_duration, epochs_per_block):
    """Used for task only: create baseline and fixed task block epochs."""
    sfreq = raw_labels.info["sfreq"]
    epoch_samples = int(round(epoch_duration * sfreq)) #converts epoch duration from seconds into samples
    events = []
    metadata = []

    for block, start_sample in block_starts:
        for epoch_in_block in range(1, epochs_per_block + 1): 
            sample = int(start_sample + (epoch_in_block - 1) * epoch_samples) # epoch 1 starts at start_sample, epoch 2 starts 5 seconds later, epoch 3 starts 10 seconds later
            events.append([sample, 0, 1]) #[sample, previous_value, event_code]
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
        event_id=None,
        tmin=0.0,
        # MNE includes tmax, so subtract one sample.
        # Example: 5 s at 1000 Hz -> 5 - 1/1000 = 4.999 s.
        tmax=epoch_duration - 1.0 / sfreq,
        baseline=None,
        metadata=pd.DataFrame(metadata),
        preload=True,
        reject=None,
        flat=None,
        reject_by_annotation=False,
        verbose="ERROR",
    )


def _make_rs_fixed_epochs(raw_labels, task_name, epoch_duration):
    """Create fixed-length RS epochs across the full recording."""
    sfreq = raw_labels.info["sfreq"]
    fixed_events = mne.make_fixed_length_events(
        raw_labels,
        id=1,
        start=0.0,
        stop=raw_labels.times[-1],
        duration=epoch_duration,
        first_samp=True,
    )
    return mne.Epochs(
        raw_labels,
        events=fixed_events,
        event_id=None,
        tmin=0.0,
        tmax=epoch_duration - 1.0 / sfreq,
        baseline=None,
        metadata=pd.DataFrame(
            {
                "condition": [f"{task_name}_fixed"] * len(fixed_events),
                "segment": [1] * len(fixed_events),
                "epoch_in_segment": list(range(1, len(fixed_events) + 1)),
                "duration_s": [epoch_duration] * len(fixed_events),
            }
        ),
        preload=True,
        reject=None,
        flat=None,
        reject_by_annotation=False,
        verbose="ERROR",
    )


def make_task_epochs(raw_labels, epoch_duration=5.0, s15_annotation="Stimulus/S 15", block_start_annotation="Stimulus/S 10"):
    """Used for task only: create baseline, fixed-task, and sequence epochs."""
    events, event_id = mne.events_from_annotations(raw_labels, verbose="ERROR") 
    print("Event IDs found:", event_id)

    s15_events, _ = _events_for_annotation(events, event_id, s15_annotation) # Get S15 events
    s10_events, _ = _events_for_annotation(events, event_id, block_start_annotation) # Get S10 events

    sfreq = raw_labels.info["sfreq"]
    task_blocks = _task_blocks_from_s15_s10(s15_events, s10_events) # Each block has block number, S15 sample, and S10 sample

    no_stim_starts = [(block["block"], block["s15"] - int(round(20.0 * sfreq))) for block in task_blocks] #
    stim_starts = [(block["block"], block["s15"]) for block in task_blocks]
    fixed_task_starts = [(block["block"], block["s10"]) for block in task_blocks]

    baseline_no_stim_epochs = _make_task_block_epochs(
        raw_labels, no_stim_starts, "baseline_no_stim", epoch_duration, 4
    )
    baseline_stim_epochs = _make_task_block_epochs(
        raw_labels, stim_starts, "baseline_stim", epoch_duration, 5
    )
    fixed_task_epochs = _make_task_block_epochs(
        raw_labels, fixed_task_starts, "task_fixed", epoch_duration, 18
    )
    sequence_epochs = make_task_sequence_epochs(
        raw_labels, task_blocks, s10_events, task_duration=90.0
    )

    return baseline_no_stim_epochs, baseline_stim_epochs, fixed_task_epochs, sequence_epochs


def make_rs_epochs(raw_labels, task_name, epoch_duration=5.0, s15_annotation="Stimulus/S 15"):
    """Used for RSpre/RSpost/RSstim only: create fixed 5 s epochs."""
    print(
        f"Creating fixed-length {task_name} epochs every {epoch_duration:g} s "
        "without using S15 annotations."
    )
    return _make_rs_fixed_epochs(raw_labels, task_name, epoch_duration)


def make_task_sequence_epochs(raw_labels, task_blocks, s10_events, task_duration=90.0):
    """Used for task only: create one variable-length epoch per task sequence."""
    sfreq = raw_labels.info["sfreq"]
    # Convert the expected task block duration from seconds to samples.
    task_samples = int(round(task_duration * sfreq))
    sequence_epochs = []

    for block in task_blocks:
        block_number = block["block"]
        task_start = block["s10"]
        task_stop = task_start + task_samples
        block_s10 = [int(event[0]) for event in s10_events if task_start <= int(event[0]) < task_stop]  # Take the sample of each S10 event from the start of this task block until the end of this task block.

        for sequence_index, start_sample in enumerate(block_s10, start=1):
            # A sequence stops at the next S10, or at the end of the task block if it is the last sequence in that block.
            if sequence_index < len(block_s10):
                stop_sample = block_s10[sequence_index]
            else:
                stop_sample = task_stop
            # This duration can differ between sequences, so each sequence is saved as a separate Epochs file.
            duration = (stop_sample - start_sample) / sfreq
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
                events=np.array([[start_sample, 0, 1]], dtype=int),
                event_id=None,
                tmin=0.0,
                # MNE includes tmax, so subtract one sample.
                # Example: 5 s at 1000 Hz -> 5 - 1/1000 = 4.999 s.
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
    epochs = epochs[np.argsort(epochs.events[:, 0])]
    epochs.save(path, overwrite=True)
    # Real duration = last time - first time + one sample. times[0]  = 0.000 times[-1] = 4.999 & 4.999 + 0.001 = 5.000 s
    duration = epochs.times[-1] - epochs.times[0] + 1.0 / epochs.info["sfreq"]
    print(f"Saved {len(epochs)} epochs ({duration:.3f} s each): {path}")
