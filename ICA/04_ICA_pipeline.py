import os
import mne
from utils import annotate_eeg, ICA_removal, ensure_task_fif_path, save_with_perms, chmod_770
from utils_mae import crop_eeg_data
mne.viz.set_browser_backend('qt', verbose=None)


# ===== User Configuration ===== #
subject = input("Enter subject number (e.g., 34): ").strip()

# ===== Session selection ===== #
session_input = input("Enter session(s) (1,2,3 or 'all'): ").strip().lower()
if session_input == "all":
    sessions = [1, 2, 3]
else:
    sessions = [int(s.strip()) for s in session_input.split(',')]

# ===== Task selection ===== #
datasets = [
    "RSpre", # Resting state pre
    "B", # Baseline 
    "task", 
    "taskv2", # Training task 
    "RSpost", # Resting_state_post
    "cTBS", # cTBS
    "RSstim"
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
data_root = f"/work/uphummel/studies/tTIS-EEG/derivatives/EEG"
output_path = f"/work/uphummel/studies/tTIS-EEG/derivatives/EEG"

# ===== Loop over Sessions and Tasks ===== ==== #
for session in sessions:
    for task_name in selected_tasks:
        task_code = task_name
        base_filename = f"sub-41Y{subject}_ses-{session}_task-{task_code}_eeg" 
        #base_filename = f"sub-41Y{subject}_ses-{session}_task-{task_code}_eeg"
        fif_path, task_dir = ensure_task_fif_path(data_root, subject, session, task_code, base_filename)
        output_dir = os.path.join(output_path, task_dir)
        os.makedirs(output_dir, mode=0o770, exist_ok=True)
        os.chmod(output_dir, 0o770)

        if not fif_path or not os.path.exists(fif_path):
            print(f"[!] File not found: {fif_path} - skipping")
            print("If error in the directory, manually put here the correct one")
            fif_path = input("Path: ")
            fif_path= fif_path.strip()
            #continue

        print(f"\n=== Processing: Session {session}, Task '{task_name}' ===")
        print(f"Loading EEG data from: {fif_path}")
        raw_filt = mne.io.read_raw_fif(fif_path, preload=True)
        print("✅ Data loaded successfully.")
        raw_filt = crop_eeg_data(raw_filt, task_name, subject_id=subject, session_id=str(session))


        # ===== Step 1: Bad trial removal by annotation ===== #
        
        annotate_eeg(raw_filt, start=0, duration=20)
        #raw_filt.save(os.path.join(output_dir, f"{base_filename}_annotated_eeg.fif"), overwrite=True) 

        # Save with permissions
        save_with_perms(raw_filt, os.path.join(output_dir, f"{base_filename}_annotated_eeg.fif"))

        print(f"✅ Done: {base_filename}_annotated_eeg.fif saved to {output_dir}" )

        # ===== Step 2: ICA ===== #

        annotated_path = os.path.join(output_dir, f"{base_filename}_annotated_eeg.fif")
        annotated_data=mne.io.read_raw_fif(annotated_path, preload=True)
        raw_filt, ica =ICA_removal(annotated_data, output_dir, base_filename)

        # ===== Step 3 and Step 4 and Step 5: Interpolate bad channels, Re-reference to average, Save final preprocessed files   ===== #

        analysis_input = input("Enter sensor or source level analysis or both: ").strip() #here the user decides if he wants to do sensor or source analysis or both

        """
        source level analysis requires EEG channels without the interpolation of bad channels
        sensor level analysis requires EEG channels with the interpolation of bad channels
        if selection is both the data will be interpolated but also the non-interpolated data will be saved
        """ 

    if analysis_input == "sensor":

        sensor_data = raw_filt.copy()
        sensor_data.interpolate_bads(reset_bads=True) # Interpolate bad channels and needs to be added for the tTIS electrodes
        sensor_data.set_eeg_reference(projection = True)  
        #sensor_data.save(os.path.join(output_dir, f"{base_filename}_final_preprocessed_raw.fif"), overwrite=True) # change here for the filename if needed, need to use later: raw.apply_proj()

        # Save with permissions
        save_with_perms(sensor_data, os.path.join(output_dir, f"{base_filename}_final_preprocessed_raw_sensor_level.fif"))
        
        print(f"✅ Done: {base_filename}_final_preprocessed_raw.fif saved to {output_dir}" ) 

    elif analysis_input == "source":

        source_data = raw_filt.copy()
        source_data.set_eeg_reference(projection = True) #set projection=True for source reconstruction
        #source_data.save(os.path.join(output_dir, f"{base_filename}_final_preprocessed_raw.fif"), overwrite=True) # change here for the filename if needed

        # Save with permissions
        save_with_perms(source_data, os.path.join(output_dir, f"{base_filename}_final_preprocessed_raw_source_level.fif"))

        
        print(f"✅ Done: {base_filename}_final_preprocessed_raw.fif saved to {output_dir}" )
        
    elif analysis_input == "both":

        sensor_data = raw_filt.copy()
        sensor_data.interpolate_bads(reset_bads=True) # Interpolate bad channels and needs to be added for the tTIS electrodes
        sensor_data.set_eeg_reference(projection = True)  
        #sensor_data.save(os.path.join(output_dir, f"{base_filename}_final_preprocessed_raw_sensor_level.fif"), overwrite=True) # change here for the filename if needed
        
        # Save with permissions
        save_with_perms(sensor_data, os.path.join(output_dir, f"{base_filename}_final_preprocessed_raw_sensor_level.fif"))
        print(f"✅ Done: {base_filename}_final_preprocessed_raw_sensor_level.fif saved to {output_dir}" ) 

        source_data = raw_filt.copy()
        source_data.set_eeg_reference(projection = True)  
        save_with_perms(source_data, os.path.join(output_dir, f"{base_filename}_final_preprocessed_raw_source_level.fif"))
        
        print(f"✅ Done: {base_filename}_final_preprocessed_raw_source_level.fif saved to {output_dir}" )

    print("\nPreprocessing completed for all selected sessions and tasks!")