"""Utility functions for 08_EEG_epoch_label_time_courses.py."""

from collections import Counter

import mne
import numpy as np


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


def _pair_block_events(start_events, end_events, sfreq):
    """Pair each block start with the next block end."""
    pairs = []
    end_index = 0

    for start_event in start_events:
        start_sample = int(start_event[0])
        while end_index < len(end_events) and int(end_events[end_index, 0]) <= start_sample:
            end_index += 1

        if end_index >= len(end_events):
            raise ValueError(
                "Could not pair every block start with a following block end. "
                f"Unpaired start at {start_sample / sfreq:.3f} s."
            )

        end_sample = int(end_events[end_index, 0])
        pairs.append((start_sample, end_sample))
        end_index += 1

    return pairs


def make_task_epochs(
    raw_labels,
    annotation_name="Stimulus/S 15",
    epoch_duration=5.0,
    n_epochs_per_block=None,
    block_start_annotation="Stimulus/S 10",
    block_end_annotation="Stimulus/S  8",
):
    """Create task baseline epochs at S15 and task epochs from S10 to S8."""
    events, event_id = mne.events_from_annotations(raw_labels, verbose="ERROR") 
    print("Event IDs found:", event_id)

    baseline_events, baseline_id = _events_for_annotation(events, event_id, annotation_name)
    block_start_events, _ = _events_for_annotation(events, event_id, block_start_annotation)
    block_end_events, _ = _events_for_annotation(events, event_id, block_end_annotation)

    if len(baseline_events) != len(block_start_events):
        raise ValueError(
            "Task baseline and block-start annotation counts differ: "
            f"{len(baseline_events)} {annotation_name!r} events, "
            f"{len(block_start_events)} {block_start_annotation!r} events."
        )

    baseline_epochs = mne.Epochs(
        raw_labels,
        events=baseline_events,
        event_id={annotation_name: baseline_id},
        tmin=-1.0,
        tmax=0.0,
        baseline=None,
        preload=True,
        reject=None,
        flat=None,
        reject_by_annotation=False,
        verbose="ERROR",
    )

    sfreq = raw_labels.info["sfreq"]
    epoch_samples = int(round(epoch_duration * sfreq))
    if epoch_samples <= 0:
        raise ValueError(f"Epoch duration must be positive, got {epoch_duration}.")

    block_pairs = _pair_block_events(block_start_events, block_end_events, sfreq)
    nonbaseline_epochs_list = []
    epochs_per_block = []

    for block_index, (start_sample, end_sample) in enumerate(block_pairs, start=1):
        if end_sample <= start_sample:
            raise ValueError(
                f"Block {block_index} end is not after start: "
                f"{start_sample / sfreq:.3f}-{end_sample / sfreq:.3f} s."
            )

        event_samples = np.arange(start_sample, end_sample - epoch_samples + 1, epoch_samples, dtype=int)
        if n_epochs_per_block is not None:
            event_samples = event_samples[:n_epochs_per_block]

        if len(event_samples) == 0:
            raise ValueError(
                f"No complete {epoch_duration:.3f} s epochs fit in block {block_index} "
                f"from {start_sample / sfreq:.3f} to {end_sample / sfreq:.3f} s."
            )

        epochs_per_block.append(len(event_samples))
        print(
            f"Block {block_index}: {len(event_samples)} epochs from "
            f"{start_sample / sfreq:.3f} to {end_sample / sfreq:.3f} s."
        )

        fixed_events = np.column_stack(
            [
                event_samples,
                np.zeros(len(event_samples), dtype=int),
                np.full(len(event_samples), 5, dtype=int),
            ]
        )
        epochs_for_block = mne.Epochs(
            raw_labels,
            events=fixed_events,
            event_id={"Custom/5": 5},  # id=5
            tmin=0.0,
            tmax=epoch_duration - 1.0 / sfreq,
            baseline=None,
            preload=True,
            reject=None,
            flat=None,
            reject_by_annotation=False,
            verbose="ERROR",
        )
        nonbaseline_epochs_list.append(epochs_for_block)

    if len(set(epochs_per_block)) != 1:
        raise ValueError(
            "Blocks yielded different numbers of complete epochs. "
            f"Epoch counts by block: {epochs_per_block}. "
            "Use an equal S10-to-S8 duration per block or set --n-epochs-per-block "
            "to truncate every block to the same count."
        )

    nonbaseline_epochs = mne.concatenate_epochs(nonbaseline_epochs_list)

    return baseline_epochs, nonbaseline_epochs


def make_rspre_epochs(raw_labels, baseline_duration=20.0, epoch_duration=5.0):
    """Create RSpre baseline from first seconds and non-baseline fixed epochs."""
    sfreq = raw_labels.info["sfreq"]
    recording_duration = raw_labels.n_times / sfreq #converts samples into seconds.
    if recording_duration <= baseline_duration:
        raise ValueError(
            f"Recording is only {recording_duration:.3f} s, shorter than "
            f"the requested baseline duration {baseline_duration:.3f} s."
        )

    baseline_events = np.array([[0, 0, 1]], dtype=int)  # [sample_number, previous_value, event_code]
    baseline_epochs = mne.Epochs(
        raw_labels,
        events=baseline_events,
        event_id={"RSpre/baseline_first20": 1},  # id=1
        tmin=0.0,
        tmax=baseline_duration,
        baseline=None,
        preload=True,
        reject=None,
        flat=None,
        reject_by_annotation=False,
        verbose="ERROR",
    )

    nonbaseline_events = mne.make_fixed_length_events(
        raw=raw_labels,
        start=baseline_duration,
        stop=recording_duration,
        duration=epoch_duration,
        id=2,
    )
    if len(nonbaseline_events) == 0:
        raise ValueError(
            f"No {epoch_duration:.3f} s non-baseline epochs fit after "
            f"the first {baseline_duration:.3f} s."
        )

    nonbaseline_epochs = mne.Epochs(
        raw_labels,
        events=nonbaseline_events,
        event_id={"RSpre/nonbaseline": 2},
        tmin=0.0,
        tmax=epoch_duration,
        baseline=None,
        preload=True,
        reject=None,
        flat=None,
        reject_by_annotation=False,
        verbose="ERROR",
    )

    return baseline_epochs, nonbaseline_epochs


def save_epochs(epochs, path):
    """Save Epochs FIF."""
    epochs.save(path, overwrite=True)
    print(f"Saved {len(epochs)} epochs: {path}")
