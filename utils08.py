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


def make_task_epochs(raw_labels, annotation_name="Stimulus/S 15", epoch_duration=5.0, n_epochs_per_block=19):
    """Create task baseline and post-S15 epochs following 05_EEG_blocks_epochs.py."""
    events, event_id = mne.events_from_annotations(raw_labels, verbose="ERROR") 
    print("Event IDs found:", event_id)

    if annotation_name not in event_id:
        raise ValueError(
            f"Could not find annotation {annotation_name!r}. "
            f"Available annotations: {sorted(event_id)}"
        )

    s15_id = event_id[annotation_name]
    # events rows are [sample_number, previous_value, event_id].
    # Keep only rows whose event_id corresponds to the S15 annotation.
    s15_events = events[events[:, 2] == s15_id]
    print(f"{annotation_name} events:")
    print(s15_events)

    if len(s15_events) == 0:
        raise ValueError(f"No events found for {annotation_name!r}.")

    baseline_epochs = mne.Epochs(
        raw_labels,
        events=s15_events,  # contains all S15 events, so this creates one baseline epoch per S15
        event_id={annotation_name: s15_id},
        tmin=-1.0,
        tmax=0.0,
        baseline=None,
        preload=True,
        reject=None,
        flat=None,
        reject_by_annotation=False,
        verbose="ERROR",
    )

    nonbaseline_epochs_list = []
    
    sfreq = raw_labels.info["sfreq"]
    for s15_event in s15_events:  # loops over the 9 tasks blocks
        start_time = s15_event[0] / sfreq
        # Creates events every epoch duration
        fixed_events = mne.make_fixed_length_events(
            raw=raw_labels,
            start=start_time,
            duration=epoch_duration,
            stop=start_time + epoch_duration * n_epochs_per_block,
            id=5,
        )
        # Create epochs for each block
        epochs_for_block = mne.Epochs(
            raw_labels,
            events=fixed_events,
            event_id={"Custom/5": 5},  # id=5
            tmin=0.0,
            tmax=epoch_duration,
            baseline=None,
            preload=True,
            reject=None,
            flat=None,
            reject_by_annotation=False,
            verbose="ERROR",
        )
        nonbaseline_epochs_list.append(epochs_for_block)

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
