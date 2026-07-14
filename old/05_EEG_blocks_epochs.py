import os
import mne
from utils import epoch_for_connectivity, baseline_epochs, annotate_eeg, ICA_removal, epoching
import matplotlib.pyplot as plt
mne.viz.set_browser_backend('qt', verbose=None)

# ===== User Configuration ===== #
user = input("Enter your Gaspar username: ").strip()
subject = input("Enter subject number (e.g., 34): ").strip()

# ===== Session selection ===== #
session_input = input("Enter session(s) (1,2,3 or 'all'): ").strip().lower()
if session_input == "all":
    sessions = [1, 2, 3]
else:
    sessions = [int(s.strip()) for s in session_input.split(',')]

# ===== Task selection ===== #
datasets = [
    "RS_pre", # Resting state pre
    "B", # Baseline 
    "task", # Training task 
    "RS_post", # Resting_state_post
    "RS_cTBS" # cTBS
]

print(f"Available tasks: {datasets}")
task_input = input("Enter task(s) (comma-separated or 'all'): ").strip()

if task_input == "all":
    selected_tasks = datasets
else:
    selected_tasks = [t.strip() for t in task_input.split(',')]
    for task in selected_tasks:
        if task not in datasets:
            raise ValueError(f"Task '{task}' not recognized. Available: {datasets}")
        
# ===== Paths and Root Setup ===== #
data_root = f"/home/{user}/mnt/Hummel-Data/TI/TI_EEG/EEG"
raw_data_path = os.path.join(data_root, "EEG_Data/Pilots_task")
preprocessed_root = os.path.join(data_root, "Preprocessed")

# ===== Loop over Sessions and Tasks ===== #
for session in sessions:
    for task_name in selected_tasks:
        task_code = task_name
        subject_dir = f"TIP{subject}/ses_{session}"
        base_filename = f"TIP{subject}_ses{session}_{task_code}"
        output_dir = os.path.join(preprocessed_root, subject_dir)
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(os.path.join(output_dir, f"{base_filename}_final_preprocessed_raw_250.fif")): 
            hd = input("How is your server mounted? (e.g. Hummel-data, Hummel-Data?) ")
            data_root = f"/home/{user}/mnt/{hd}/TI/TI_EEG/EEG"
            raw_data_path = os.path.join(data_root, "EEG_Data/Pilots_task")
            preprocessed_root = os.path.join(data_root, "Preprocessed")
            output_dir = os.path.join(preprocessed_root, subject_dir)
            os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(os.path.join(output_dir, f"{base_filename}_final_preprocessed_raw_250.fif")):
            print(f"[!] Folder not found: {output_dir}.")
            print("If error in the directory, manually put here the correct one")
            output_dir = input("Path: ")
            output_dir= output_dir.strip()
            #continue

        print(f"\n=== Processing: Session {session}, Task '{task_name}' ===")
        print(f"Loading EEG data from: {base_filename}")
        filtered_data = mne.io.read_raw_fif(os.path.join(output_dir, f"{base_filename}_final_preprocessed_raw_250.fif"), preload=True)
        #fif_path = os.path.join(output_dir, f"{base_filename}_filt_raw.fif") #UNCOMMENT THAT IF YOU WANT TO RERUN ICA IN BLOCKS
        #filtered_data = mne.io.read_raw_fif(filtered_data, preload=True)

        #filtered_data.plot(scalings=dict(eeg=10e-5), block=True)

        # ===== Step 1: Cropping data ===== #
        events, event_id = mne.events_from_annotations(filtered_data)

        s15_id = event_id['Stimulus/S 15']  
        s15_events = events[events[:, 2] == s15_id]
        print(s15_events)

        sfreq = filtered_data.info['sfreq']

        blocks = []

        for i, event in enumerate(s15_events):
            onset_sample = event[0]
            onset_time = onset_sample / sfreq

            start = onset_time -1
            end = onset_time + 95

        # Check that cropping won't go out of bounds
            if start < 0 or end > filtered_data.times[-1] - 0.001:
                print(f"⚠️ Skipping event {i+1}: crop range out of bounds ({start:.2f}s to {end:.2f}s)")
                continue

            raw_segment = filtered_data.copy().crop(tmin=start, tmax=end)

            #raw_segment.plot(scalings=dict(eeg=10e-5), block=True)
            #print(f"✅ Cropped segment {i+1}: {start:.2f}s to {end:.2f}s")


            blocks.append(raw_segment) #List of 9 raw objects
            print(f"✅ Cropped block {i+1}: {start:.2f}s to {end:.2f}s")

            print(f"\n🎉 Total cropped blocks: {len(blocks)}")

        # Concatenate all blocks into a single raw object
        if len(blocks) > 0:
            concatenated_blocks = mne.concatenate_raws(blocks)
            #concatenated_blocks.plot(scalings=dict(eeg=10e-5), block=True)
            print(f"✅ Concatenated {len(blocks)} blocks into a single raw object.")

            concatenated_blocks.save(os.path.join(output_dir, f"{base_filename}_blocks_raw_250.fif"), overwrite=True)
            print(f"✅ Done: {base_filename}_blocks_raw.fif saved to {output_dir}" ) #LIST of raw objects

"""
        # ===== Bad trial removal by annotation ===== #
    
        annotate_eeg(concatenated_blocks, start=0, duration=20)

        # ===== Step 2: ICA ===== #
        ICA_removal(concatenated_blocks, output_dir, base_filename)

        # ===== Step 3: Interpolate bad channels ===== #
        concatenated_blocks.interpolate_bads(reset_bads=True) # Interpolate bad channels
        
        # ===== Step 4: Re-reference to average ===== #
        concatenated_blocks.set_eeg_reference(projection = True) #set projection=True for source reconstruction # SC@17/4 added argument to set projection to True

        # ===== Step 5: Save final preprocessed file ===== #
        concatenated_blocks.save(os.path.join(output_dir, f"{base_filename}_final_concatenated_blocks_preprocessed_raw.fif"), overwrite=True)
        print(f"✅ Done: {base_filename}_final__concatenated_blocks_preprocessed_raw.fif saved to {output_dir}" )

# Epochs_connectivity 
epochs_for_connectivity = epoch_for_connectivity(concatenated_blocks, duration=5.0)

print("Number of Epochs before dropping ramp-up", len(epochs_for_connectivity))
epochs_for_connectivity.plot(scalings=dict(eeg=10e-5), block=True)

#to_drop = input("Enter epochs which you want to drop due to ramp-up: ").strip()
#print(f"Dropping ramp-up epochs at indices: {to_drop}")
#epochs_for_connectivity.drop(to_drop)

#print("Number of Epochs after dropping ramp-up", len(epochs_for_connectivity))

print("✅ Number of epochs for connectivity: ", len(epochs_for_connectivity))

epochs_for_connectivity.save(os.path.join(output_dir, f"{base_filename}_epochs_connectivity-epo.fif"), overwrite=True)
print(f"✅ Done: {base_filename}_epochs_connectivity-epo.fif saved to {output_dir}" )

#Epochs_baseline
epochs_for_baseline = baseline_epochs(concatenated_blocks)
epochs_for_baseline.save(os.path.join(output_dir, f"{base_filename}_epochs_baseline-epo.fif"), overwrite=True)
print(f"✅ Done: {base_filename}_epochs_baseline-epo.fif saved to {output_dir}" )


epochs_key_presses = epoching(concatenated_blocks, events, event_id, -0.1, 0.5)

print("Number of Epochs before dropping ramp-up", len(epochs_key_presses))
epochs_key_presses.plot(scalings=dict(eeg=10e-5), block=True)
epochs_key_presses.save(os.path.join(output_dir, f"{base_filename}_epochs_key_presses-epo.fif"), overwrite=True)
print(f"✅ Done: {base_filename}_epochs_key_presses-epo.fif saved to {output_dir}" )

"""