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


def _sample_to_time(sample, sfreq):
    """Convert an event sample index to seconds."""
    return int(sample) / sfreq


def _pair_task_blocks(s15_events, s10_events, s8_events, sfreq):
    """Pair task blocks as S15-20s -> S15 -> first S10 -> S8."""
    print(f"Event counts for task pairing: {len(s15_events)} S15, {len(s10_events)} S10, {len(s8_events)} S8.")
    if len(s15_events) != len(s8_events):
        raise ValueError(f"Expected the same number of S15 and S8 events, got {len(s15_events)} S15 and {len(s8_events)} S8.")

    blocks = []
    no_stim_samples = int(round(20.0 * sfreq))

    # 1st value of block index is start=1
    for block_index, s15_event in enumerate(s15_events, start=1):
        s15_sample = int(s15_event[0])
        s8_sample = int(s8_events[block_index - 1, 0])
        no_stim_start_sample = s15_sample - no_stim_samples
        if no_stim_start_sample < 0:
            raise ValueError(f"Cannot create 20 s no-stim baseline before S15 for block {block_index}: S15={_sample_to_time(s15_sample, sfreq):.3f} s.")
        if not (s15_sample < s8_sample):
            raise ValueError(f"Expected S15 < S8 for block {block_index}: S15={_sample_to_time(s15_sample, sfreq):.3f} s, S8={_sample_to_time(s8_sample, sfreq):.3f} s.")

        s10_matches = s10_events[(s10_events[:, 0] > s15_sample) & (s10_events[:, 0] < s8_sample)]
        if len(s10_matches) == 0:
            raise ValueError(f"Could not find S10 between S15 at {_sample_to_time(s15_sample, sfreq):.3f} s and S8 at {_sample_to_time(s8_sample, sfreq):.3f} s.")
        s10_sample = int(s10_matches[0, 0]) # 1st row 1st column 
        blocks.append({"no_stim_start": no_stim_start_sample, "s15": s15_sample, "s10": s10_sample, "s8": s8_sample})

    return blocks


def _make_interval_epochs(raw_labels, intervals, event_name, event_id, sfreq):
    """Create one epoch per interval, requiring equal interval durations."""
    durations = np.array([stop - start for start, stop in intervals], dtype=int)
    if np.any(durations <= 0):
        raise ValueError(f"{event_name} contains an empty or negative interval.")
    if len(set(durations)) != 1:
        durations_seconds = [duration / sfreq for duration in durations]
        raise ValueError(
            f"{event_name} intervals have different durations: {durations_seconds}. "
            "MNE Epochs require equal-length intervals."
        )

    epoch_events = np.column_stack([[start for start, _ in intervals], np.zeros(len(intervals), dtype=int), np.full(len(intervals), event_id, dtype=int)]) #[sample_number, previous_event_value, event_id]
    
    duration = int(durations[0])
    return mne.Epochs(
        raw_labels,
        events=epoch_events,
        event_id={event_name: event_id},
        tmin=0.0,
        tmax=(duration - 1) / sfreq,
        baseline=None,
        preload=True,
        reject=None,
        flat=None,
        reject_by_annotation=False,
        verbose="ERROR",
    )

def make_task_epochs(
    raw_labels,
    epoch_duration=5.0,
    n_epochs_per_block=None,
    s15_annotation="Stimulus/S 15",
    block_start_annotation="Stimulus/S 10",
    block_end_annotation="Stimulus/S  8",
):
    """Create S15-20s to S15, S15-S10, and S10-S8 task epochs."""
    events, event_id = mne.events_from_annotations(raw_labels, verbose="ERROR") 
    print("Event IDs found:", event_id)

    s15_events, _ = _events_for_annotation(events, event_id, s15_annotation)
    s10_events, _ = _events_for_annotation(events, event_id, block_start_annotation)
    s8_events, _ = _events_for_annotation(events, event_id, block_end_annotation)

    sfreq = raw_labels.info["sfreq"]
    epoch_samples = int(round(epoch_duration * sfreq))
    if epoch_samples <= 0:
        raise ValueError(f"Epoch duration must be positive, got {epoch_duration}.")

    task_blocks = _pair_task_blocks(s15_events, s10_events, s8_events, sfreq)
    no_stim_intervals = [(block["no_stim_start"], block["s15"]) for block in task_blocks]
    stim_intervals = [(block["s15"], block["s10"]) for block in task_blocks]

    baseline_no_stim_epochs = _make_interval_epochs(raw_labels, no_stim_intervals, "baseline/no_stim/S4-S15", 1, sfreq)
    baseline_stim_epochs = _make_interval_epochs(raw_labels, stim_intervals, "baseline/stim/S15-S10", 2, sfreq)

    nonbaseline_epochs_list = []
    epochs_per_block = []

    for block_index, block in enumerate(task_blocks, start=1):
        start_sample = block["s10"]
        end_sample = block["s8"]
        if end_sample <= start_sample:
            raise ValueError(f"Block {block_index} end is not after start: {start_sample / sfreq:.3f}-{end_sample / sfreq:.3f} s.")

        event_samples = np.arange(start_sample, end_sample - epoch_samples + 1, epoch_samples, dtype=int)
        if n_epochs_per_block is not None:
            event_samples = event_samples[:n_epochs_per_block]

        if len(event_samples) == 0:
            raise ValueError(f"No complete {epoch_duration:.3f} s epochs fit in block {block_index} from {start_sample / sfreq:.3f} to {end_sample / sfreq:.3f} s.")

        epochs_per_block.append(len(event_samples))
        print(f"Block {block_index}: {len(event_samples)} epochs from {start_sample / sfreq:.3f} to {end_sample / sfreq:.3f} s.")

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
        raise ValueError(f"Blocks yielded different numbers of complete epochs. Epoch counts by block: {epochs_per_block}. Use an equal S10-to-S8 duration per block or set --n-epochs-per-block to truncate every block to the same count.")

    nonbaseline_epochs = mne.concatenate_epochs(nonbaseline_epochs_list)

    return baseline_no_stim_epochs, baseline_stim_epochs, nonbaseline_epochs


def make_rspre_epochs(raw_labels, baseline_duration=20.0, epoch_duration=5.0):
    """Create RSpre baseline from first seconds and non-baseline fixed epochs."""
    sfreq = raw_labels.info["sfreq"]
    recording_duration = raw_labels.n_times / sfreq #converts samples into seconds.
    if recording_duration <= baseline_duration:
        raise ValueError(f"Recording is only {recording_duration:.3f} s, shorter than the requested baseline duration {baseline_duration:.3f} s.")

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
        raise ValueError(f"No {epoch_duration:.3f} s non-baseline epochs fit after the first {baseline_duration:.3f} s.")

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
