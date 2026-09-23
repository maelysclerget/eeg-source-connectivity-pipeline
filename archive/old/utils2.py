import matplotlib.pyplot as plt
import os
import pyprep
import numpy as np
import mne

def epoching(filtered_data, event, event_id, tmin, tmax, baseline_start, baseline_end):
    epoched_data = mne.Epochs(filtered_data, events=event, event_id=event_id, tmin=tmin, tmax=tmax, baseline=(baseline_start, baseline_end), reject_by_annotation=False)
    epoched_data.plot(scalings=dict(eeg=10e-5), block=True)

"""
def tfr():

def apply_baseline_correction_epochs_from_raw(epochs_to_correct, baseline_raw, baseline_interval):

    Apply baseline correction to an Epochs object using a Raw object as the baseline source.

    Parameters:
    - epochs_to_correct: Epochs object (MNE) to be baseline corrected.
    - baseline_raw: Raw object (MNE) providing the baseline data.
    - baseline_interval: Tuple (tmin, tmax) defining the time interval for baseline calculation (in seconds).

    Returns:
    - baseline_corrected_epochs: A copy of the Epochs object with baseline-corrected data.

    
    # Crop raw data to the baseline interval
    baseline_data = baseline_raw.copy().crop(tmin=baseline_interval[0], tmax=baseline_interval[1])
    
    # Get baseline data as (n_channels, n_times)
    baseline_array = baseline_data.get_data()

    # Compute channel-wise mean and std over time
    baseline_avg = baseline_array.mean(axis=1)  # (n_channels,)
    baseline_std = baseline_array.std(axis=1)   # (n_channels,)
    
    # Get epochs data: (n_epochs, n_channels, n_times)
    epochs_data = epochs_to_correct.get_data()
    
    # Apply baseline correction: z-score across time using raw baseline
    corrected_data = (epochs_data - baseline_avg[np.newaxis, :, np.newaxis]) / baseline_std[np.newaxis, :, np.newaxis]
    
    # Copy the original epochs and replace data
    corrected_epochs = epochs_to_correct.copy()
    corrected_epochs._data = corrected_data

    return corrected_epochs

    """