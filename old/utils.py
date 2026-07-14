import matplotlib.pyplot as plt
import os
import pyprep
import numpy as np
import mne
from mne.preprocessing import ICA
mne.viz.set_browser_backend('qt', verbose=None)

def annotate_eeg(raw_data, start=0, duration=20):
    """
    Plots raw EEG data, allows manual annotation of bad segments and channels. Bad segments will automatically be ignored in subsequent sections of the pipeline, such as ICA and Epoching.

    Parameters:
    - raw_data: mne.io.Raw
    - start: float, start time of the plot
    - duration: float, duration (in seconds) to display

    Returns:
    - Void
    """
    fig=raw_data.plot(start=start, duration=duration, scalings=dict(eeg=10e-5), block=True)

    
    return None

def plot_impedances(vhdr_filename):
    """
    Reads and prints impedance information from a .vhdr file. Useful for reviewing electrode contact quality prior to preprocessing.

    Parameters:
    - vhdr_filename: str, path to the .vhdr file containing impedance values

    Returns:
    - Void
    """
    with open(vhdr_filename, 'r') as file:
        contents = file.read()

    print(contents)

    return None


def bad_channel_inspection(raw_data):
    """
    Plots the Power Spectral Density (PSD) of EEG channels to help identify bad channels visually. Users can input additional bad channels after inspection.

    Parameters:
    - raw_data: mne.io.Raw

    Returns:
    - Void (updates raw_data.info["bads"] in-place)
    """

    print("Computing PSD, might take some time...")
    psd = raw_data.compute_psd(method='welch', fmin=0, fmax=200, n_fft=len(raw_data.times), 
        n_overlap=0, window='hann', exclude="bads", reject_by_annotation=False)

    psd.plot()
    plt.show(block=True)

    while True:
        try:
            additionnal_bads = input("Specify any additionnal bad channels from PSD inspection (IMPORTANT use ';' between channels, e.g: Cz;FC1;Pz): ").split(';')
            if additionnal_bads != ['']:
                raw_data.info["bads"] += additionnal_bads
                print('Channels', additionnal_bads, 'added to bad channels')
            break
        except ValueError:
            print("Incorrect channel names, please specify them again")
    return None

def PREP(raw_data):
    """
    Automatically detects bad EEG channels using the PREP pipeline's noisy channel detection algorithm. Results are added to the raw data's 'bads' list.

    Parameters:
    - raw_data: mne.io.Raw

    Returns:
    - Void (updates raw_data.info["bads"] in-place)
    """
    # Initialize NoisyChannels object
    noisy_detector = pyprep.find_noisy_channels.NoisyChannels(raw_data)

    noisy_detector.find_all_bads(ransac=True, channel_wise=False, max_chunk_size=None)
    
    bad_channels_prev = set(raw_data.info["bads"])
    bad_channels_noisy = noisy_detector.get_bads()

    print(f"Bad channels pyprep: {list(bad_channels_noisy)}")
    
    all_bad_channels = set(bad_channels_prev).union(set(bad_channels_noisy))

    raw_data.info["bads"] = list(all_bad_channels)
    return None


def PSD_TI_artefacts(raw_data, output_directory, condition):
    """
    Computes and saves Power Spectral Density (PSD) plots for each EEG channel
    at multiple frequency ranges (250, 2500, 25000 Hz). PSDs are saved as PNG
    images in a dedicated 'PSDs' folder inside the specified output directory.

    Parameters:
    - raw_data: mne.io.Raw
        The raw EEG data to compute PSDs from.
    - output_directory: str
        Directory where the PSD plots will be saved.

    Returns:
    - Void
    """
    raw_hfiltered = raw_data.copy().filter(l_freq=0.5, h_freq=None)
    ch_names = raw_hfiltered.ch_names

    ffmin = 0.5
    ffmax_list = [250, 2500, 25000]

    # Create PSDs folder
    psd_dir = os.path.join(output_directory, "PSDs")
    os.makedirs(psd_dir, exist_ok=True)

    for ffmax in ffmax_list:
        for channel_name in ch_names:
            try:
                #print(f'Computing PSD for channel {channel_name}')
                # Isolate channel
                channel_data = raw_hfiltered.copy().pick_channels([channel_name])
                n_times = channel_data.n_times

                # Compute PSD
                psd_channel = channel_data.compute_psd(
                    method='welch',
                    fmin=ffmin,
                    fmax=ffmax,
                    n_fft=n_times,
                    n_overlap=0,
                    window='hann', 
                    reject_by_annotation=False,

                )

                # Plot and save
                fig = psd_channel.plot(show=False)  
                plot_filename = f'psd_{condition}_{channel_name}_max{ffmax}.png'
                fig_path = os.path.join(psd_dir, plot_filename)
                fig.savefig(fig_path)
                plt.close(fig)  
                

            except Exception as err:
                print(f"[!] Error processing channel {channel_name}: {err}")



def correct_phase_delay(raw_data):
    """
    Corrects the event's sample point to account for the 8 ms phase delay cause by the low-pass filter.

    Parameters:
    - raw_data: mne.io.Raw

    Returns:
    - Void (updates events in-place)
    """
    for i, event in enumerate(raw_data.annotations.description):
        if event=='Response/R 15' or event=='Stimulus/S 15':
            raw_data.annotations.onset[i]+=0.008


def filter_and_ds(raw_data, l_freq, h_freq, sfreq=1000):
    """
    bandpass filter, resampling and notch filter applied

    Parameters:
    - raw_data: mne.io.Raw
    - l_freq: float, cutoff freq for high-pass filter
    - h_freq: float, cutoff freq for low-pass filter


    Returns:
    - Void (updates events in-place)
    """
    # Bandpass filter
    b_filtered=raw_data.filter(l_freq=l_freq, h_freq=h_freq, n_jobs = 1) # SC@17/4 n_jobs=20 for parallel processing, can be set to -1 to use all cores

    # b_filtered_psd =b_filtered.compute_psd(
    #                 method='welch',
    #                 fmin=l_freq,
    #                 fmax=250,
    #                 n_fft=len(b_filtered.times),
    #                 n_overlap=0,
    #                 window='hann', 
    #                 reject_by_annotation=True)
    
    # b_filtered_psd.plot()
    # plt.show(block=True)

    # Downsample to 1000 Hz
    
    b_filtered =b_filtered.copy().resample(sfreq=sfreq)

    b_filtered= b_filtered.notch_filter(freqs=[50])

    # b_filtered_psd =b_filtered.compute_psd(
    #                 method='welch',
    #                 fmin=l_freq,
    #                 fmax=250,
    #                 n_fft=len(b_filtered.times),
    #                 n_overlap=0,
    #                 window='hann', 
    #                 reject_by_annotation=True)
    
    # b_filtered_psd.plot()
    # plt.show(block=True)


    return b_filtered

def ICA_removal(data, save_path, base_filename):
    """
    Applies Independent Component Analysis (ICA) to the epoched EEG data for artifact removal.
    Allows manual exclusion of components based on visual inspection of source time series and topographies.
    A second ICA is run on a copy of the data to verify the removal effect.

    Parameters:
    - epoched_data: mne.Epochs, preprocessed EEG data in epochs

    Returns:
    - Void (modifies ICA state and shows plots for manual intervention)
    """
    n_comp = len(data.info.ch_names) - len(data.info["bads"]) - 1 
    ica = ICA(n_components=n_comp, max_iter="auto", random_state=97)
    ica.fit(data)
    ica.save(os.path.join(save_path, f"{base_filename}_concatenated_blocks_before_rejection-ica_250.fif"), overwrite=True)

    ica.plot_sources(data, block=True)     
    ica.plot_components()
    
    cmp_for_properties = input("Please indicate the ICA components you want to inspect in detail: ").strip()
    cmp_for_properties = np.array([cmp.strip() for cmp in cmp_for_properties.split(',')]).astype(int) if cmp_for_properties else None
    mne.viz.plot_ica_properties(ica, data, picks=cmp_for_properties)

    cmp_to_exclude = input("Please indicate the ICA components you want to exclude: ").strip()
    cmp_to_exclude = np.array([cmp.strip() for cmp in cmp_to_exclude.split(',')]).astype(int) if cmp_to_exclude else []
    ica.exclude = cmp_to_exclude
    print(f"Excluded {len(ica.exclude)} components: {cmp_to_exclude}")
    
    ica.apply(data)

    data.plot(scalings=dict(eeg=10e-5), block=True)
    """
    data_new = data.copy()

    print("Recomputing ICA to observe component rejection effects...")
    ica2 = ICA(n_components=len(data_new.info.ch_names) - len(data_new.info["bads"]) - len(ica.exclude) - 1,
               max_iter="auto", random_state=97) # SC@17/4 minus len(ica.exclude) to the n_components

    ica2.plot_sources(data_new, block=True)
    ica2.plot_components() 
    """
    
def compute_ICA(data, n_components=None, random_state=97):
    """Fits ICA to the data and returns the ICA object without saving."""
    if n_components is None:
        n_components = len(data.info.ch_names) - len(data.info["bads"]) - 1

    ica = ICA(n_components=n_components, max_iter="auto", random_state=random_state)
    ica.fit(data)
    print(f"ICA computed with {n_components} components.")
    return ica


def save_ICA(ica, save_path, filename):
    """Saves the ICA object to disk."""
    ica_path = os.path.join(save_path, filename)
    ica.save(ica_path, overwrite=True)
    print(f"ICA saved to {ica_path}")


def ICA_visual_inspection(ica, data):
    """Plots ICA sources, components, and properties without modifying data."""
    ica.plot_sources(data, block=True)
    ica.plot_components()

    cmp_for_properties = input("Enter ICA components to inspect in detail (comma-separated): ").strip()
    if cmp_for_properties:
        picks = [int(c.strip()) for c in cmp_for_properties.split(",")]
        mne.viz.plot_ica_properties(ica, data, picks=picks)

def exclude_ICA_components(ica, data):
    """Allows user to exclude ICA components based on visual inspection."""
    cmp_to_exclude = input(f"Current excluded components: {ica.exclude}. \n Enter the updated list of ICA components to exclude (comma-separated): ").strip()
    
    if cmp_to_exclude:
        exclude = [int(c.strip()) for c in cmp_to_exclude.split(",")]
        ica.exclude = exclude
        print(f"Excluding {len(ica.exclude)} components: {exclude}")
    else:
        ica.exclude = []
        
    print(f"The following components are excluded: {ica.exclude}")
    
    return ica

def apply_ICA(ica, data):
    """Applies ICA exclusion to the data."""
    
    ica.apply(data)
    data.plot(scalings=dict(eeg=10e-5), block=True)
    
    print("ICA applied to data.")

    
def epoching(filtered_data, events, event_id, tmin, tmax):

    events, event_id = mne.events_from_annotations(filtered_data)
    print(event_id)
    print("Event IDs found:", event_id)

    event_id_dict = {'Stimulus/S  15': event_id['Stimulus/S  15']}
    
    epoched_data = mne.Epochs(
        filtered_data,
        events=events,
        event_id=event_id_dict,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        reject_by_annotation=False,
        preload=True
    )

    epoched_data.plot(scalings=dict(eeg=10e-5), block=True)
    return epoched_data    

def epoch_for_connectivity(concatenated_blocks, duration=5.0): #id =5
        
    events, event_id = mne.events_from_annotations(concatenated_blocks)
    print(event_id)
    print("Event IDs found:", event_id)

    s15_id = event_id['Stimulus/S 15']  
    s15_events = events[events[:, 2] == s15_id]
    print(s15_events)

    sfreq = concatenated_blocks.info['sfreq']
    
    all_epochs = []

    for idx, s15 in enumerate(s15_events):
        start_sample = s15[0]
        start_time = start_sample / sfreq

    #start_sample = s15_events[0, 0]  # sample index of first S15
    #start_time = start_sample / concatenated_blocks.info['sfreq']  # in seconds

    #print(f"First S15 trigger found at {start_time:.2f} seconds.")
    
        fixed_events = mne.make_fixed_length_events(
            raw=concatenated_blocks,
            start=start_time,
            duration=5.0,
            stop=start_time + duration * 19,  
            id=5)

        print("Fixed events for this block:", fixed_events)

        epochs_for_connectivity = mne.Epochs(
            concatenated_blocks,
            events=fixed_events,
            event_id={'Custom/5': 5},
            tmin=0,
            tmax=5.0,
            baseline=None,
            preload=True,
            reject=None,       
            flat=None,         
            reject_by_annotation=False
            )
        
        all_epochs.append(epochs_for_connectivity)

    epochs_for_connectivity = mne.concatenate_epochs(all_epochs)

    return epochs_for_connectivity

def baseline_epochs(concatenated_blocks): 

    events, event_id = mne.events_from_annotations(concatenated_blocks)

    print("Event IDs found:", event_id)

    s15_id = event_id['Stimulus/S 15']  
    s15_events = events[events[:, 2] == s15_id]
    print("S15 events:\n", s15_events)

    start_sample = s15_events[0, 0]  # sample index of first S15
    start_time = start_sample / concatenated_blocks.info['sfreq']  # in seconds

    print(f"First S15 trigger found at {start_time:.2f} seconds.")

    epochs_for_baseline = mne.Epochs(
        concatenated_blocks,
        events=s15_events,
        event_id={'Stimulus/S 15': s15_id},
        tmin=-1,
        tmax=0,
        baseline=None,
        preload=True,
        reject=None,       
        flat=None,         
        reject_by_annotation=False
    )

    print("✅ Number of baseline epochs: ", len(epochs_for_baseline))
    epochs_for_baseline.plot(scalings=dict(eeg=10e-5), block=True)
    return epochs_for_baseline