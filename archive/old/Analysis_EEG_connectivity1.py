import os
import mne
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import epoch_for_connectivity #tfr, apply_baseline_correction_epochs_from_raw 
from mne.io import read_raw_fif
from mne_connectivity import spectral_connectivity_epochs
from mne_connectivity.viz import plot_sensors_connectivity
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
        save_dir = os.path.join(data_root, "Processed", subject_dir)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

        if not os.path.exists(os.path.join(output_dir, f"{base_filename}_epochs_connectivity-epo.fif")): 
            hd = input("How is your server mounted? (e.g. Hummel-data, Hummel-Data?) ")
            data_root = f"/home/{user}/mnt/{hd}/TI/TI_EEG/EEG"
            raw_data_path = os.path.join(data_root, "EEG_Data/Pilots_task")
            preprocessed_root = os.path.join(data_root, "Preprocessed")
            output_dir = os.path.join(preprocessed_root, subject_dir)
            os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(os.path.join(output_dir, f"{base_filename}_epochs_connectivity-epo.fif")):
            print(f"[!] Folder not found: {output_dir}.")
            print("If error in the directory, manually put here the correct one")
            output_dir = input("Path: ")
            output_dir= output_dir.strip()
            #continue

        print(f"\n=== Processing: Session {session}, Task '{task_name}' ===")
        print(f"Loading EEG data from: {base_filename}")

        #filtered_data = mne.io.read_raw_fif(output_dir, f"{base_filename}_final_preprocessed_raw.fif", preload=True)
        fif_path = os.path.join(output_dir, f"{base_filename}_epochs_connectivity-epo.fif")
        epochs = mne.read_epochs(fif_path, preload=True) 
        #Remember here you have the epochs for all the blocks together after dropping the ramp-up epochs
        #and preserving the indices

        #Load the baseline epoch object and wish you good luck that it will work
        baseline_epochs = mne.read_epochs(os.path.join(output_dir, f"{base_filename}_epochs_baseline-epo.fif"), preload=True)


        #Drop Iz channel from the analysis, it was not set-up for 1st experiement
        epochs.drop_channels(['Iz'])
        baseline_epochs.drop_channels(['Iz'])

        print("Plotting task epochs...")
        #epochs.plot(scalings=dict(eeg=10e-5), block=True)

        print("Plotting baseline epochs...")
        #baseline_epochs.plot(scalings=dict(eeg=10e-5), block=True)

        ffmin = input("Enter the min frequency for spectral connectiivty analysis : ").strip()
        ffmax = input("Enter the max frequency for spectral connectiivty analysis : ").strip()

        #Group the epochs by blocks, to be optimised later
        epochs_block = [
                list(range(0, 18)),
                list(range(18, 36)),
                list(range(36, 54)),
                list(range(54, 72)),
                list(range(72, 90)),
                list(range(90, 108)),
                list(range(108, 126)),
                list(range(126, 144)),
                list(range(144, 161))
            ]
        
        regions = {
            "frontal": ["Fp1", "Fp2", "Fz", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "AF3", "AF4", "AF7", "AF8", "Fpz"],
            "central": ["Cz", "C1", "C2", "C3", "C4", "C5", "C6", "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "CPz", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6"],
            "parietal": ["Pz", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"],
            "temporal": ["T7", "T8", "TP7", "TP8", "TP9", "TP10", "FT7", "FT8", "FT9", "FT10"],
            "occipital": ["Oz", "O1", "O2", "PO3", "PO4", "PO7", "PO8", "PO9", "PO10", "POz", "Iz"]
            }
        
        #Compute the connectivity for baseline epochs 
        print(f"Computing iCoh for baseline epochs...")  
        baseline_con = spectral_connectivity_epochs(
                baseline_epochs,
                method='imcoh',
                mode='multitaper',
                fmin=ffmin,
                fmax=ffmax,
                faverage=True,
                sfreq=epochs.info['sfreq'],
                verbose=False
                )
        
        baseline_path = os.path.join(save_dir, f"{base_filename}_iCoh_{ffmin}_{ffmax}_baseline.fif")
        baseline_con.save(baseline_path)

        dense_baseline_con = baseline_con.get_data(output='dense')
        ##Here you need to make the baseline icoh absolute value --> Shiu-Cheng
        print("✅ Done: Shape of baseline connectivity matrix: ", dense_baseline_con.shape)

        #Compute the connectivity for task epochs
        all_con = []
        con_values = []
        channel_names = epochs.ch_names

        for i in range(9):  # Loop over each block 0 to 8
            print(f"Computing iCoh for epochs in block {i}...")

            # === Get task epochs (remaining 9) === #
            task_epochs = epochs[epochs_block[i][0:]].get_data(picks="eeg")
            print(task_epochs.shape)

            task_con = spectral_connectivity_epochs(
                task_epochs,
                method='imcoh',
                mode='multitaper',
                fmin=ffmin,
                fmax=ffmax,
                faverage=True,
                sfreq=epochs.info['sfreq'],
                verbose=False
                )

            task_path = os.path.join(save_dir, f"{base_filename}_iCoh_{ffmin}_{ffmax}_task_block_{i}.fif")
            task_con.save(task_path)

            dense_task_con = task_con.get_data(output='dense')[:, :, 0]
            ##Here you need to make the task icoh absolute value --> Shiu-Cheng
            print("Shape of dense connectivityy matrix: ", dense_task_con.shape)

            #Baseline correction using the first epoch of the block which is the baseline 
            baseline_for_block = dense_baseline_con[0, :, :] # < -- if you want CORRECTION WITH FIRST BASELINE FOR ALL BLOCKS, replace i with 0
            dense_con = dense_task_con - baseline_for_block 
            #dense_con = dense_task_con <-- if you want NO CORRECTION DO THIS HERE
            
            print(f"Plotting dense connectivity matrix for block {i}...")
            
            fig=plt.figure()
            im = plt.imshow(dense_con)

            fig.colorbar(im)
            plt.title(f"Connectivity matrix block{i} for {ffmin}-{ffmax} Hz")
            plt.xlabel("Channels")
            plt.ylabel("Channels")

            save_path = os.path.join(save_dir, f"{base_filename}_iCoh_{ffmin}_{ffmax}_connectivity_block_{i}.png")
            
            plt.savefig(save_path)
            print(f"✅ Done: {base_filename}_iCoh_{ffmin}_{ffmax}_connectivity_block_{i}.png saved to {save_path}" )
            
            #plt.show()
            plt.close()

            all_con.append(dense_con) #Append the list with the dense connectivity matrix for each block
            print(f"Done with epochs group {i}, shape: {dense_con.shape}")

            min_val = dense_con.min()
            max_val = dense_con.max()
            mean_val = dense_con[dense_con != 0].mean()
            std_val = dense_con[dense_con != 0].std()

            con_values.append([i, min_val, max_val, mean_val, std_val])

            print(f"Min iCoh {ffmin}_{ffmax} value for {i}th block:", min_val)
            print(f"Max iCoh {ffmin}_{ffmax} value {i}th block:", max_val)
            print(f"Mean iCoh {ffmin}_{ffmax} value {i}th block:", mean_val)
            print(f"SD iCoh {ffmin}_{ffmax} value {i}th block:", std_val)

            # ========== SAVE INDIVIDUAL BLOCK HEATMAP WITH REGIONS ========== #
            # ---- Region definitions done before loop ---- #
            # ---- Build region-ordered channel list ---- #
            ordered_channels = []
            region_boundaries = []
            region_names = []

            df = pd.DataFrame({
                "Ch1": [channel_names[i] for i in range(61) for j in range(61)],
                "Ch2": [channel_names[j] for i in range(61) for j in range(61)],
                "Value": dense_con.flatten()
                })

            pivot_df = df.pivot(index="Ch1", columns="Ch2", values="Value").fillna(0)

            for region, chs in regions.items():
                valid_chs = [ch for ch in chs if ch in pivot_df.index]
                ordered_channels.extend(valid_chs)
                region_boundaries.append(len(ordered_channels))  # for separators
                region_names.append((region, len(ordered_channels) - len(valid_chs) + len(valid_chs)//2))  # for labels

            pivot_ordered = pivot_df.loc[ordered_channels, ordered_channels]
            mask = pivot_ordered.values == 0

            # ---- Plot heatmap ---- #
            plt.figure(figsize=(20, 20))

            ax = sns.heatmap(
                pivot_ordered,
                cmap="coolwarm",
                center=0,
                linewidths=0.5,
                square=True,
                xticklabels=ordered_channels,
                yticklabels=ordered_channels,
                mask=mask,
                cbar_kws={"label": "Imaginary part of Coherence"}
            )

            # ---- Add region separators ---- #
            for boundary in region_boundaries[:-1]:
                ax.axhline(boundary, color='black', linewidth=1.5)
                ax.axvline(boundary, color='black', linewidth=1.5)

            for region, pos in region_names:
    
                ax.text(pos, -2, region.upper(), ha='center', va='bottom', fontsize=12, fontweight='bold', rotation=0)
                ax.text(-2, pos, region.upper(), ha='right', va='center', fontsize=12, fontweight='bold', rotation=90)

            # ---- Final formatting ---- #
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.set_label_position('top')
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            plt.title("Connectivity Matrix Channel-Channel Grouped by Brain Region", pad=100)
            plt.tight_layout(rect=[0, 0, 0.95, 1])

            summary_fig_path = os.path.join(save_dir, f"{base_filename}_iCoh_{ffmin}_{ffmax}__block_{i}_grid.png")
            plt.savefig(summary_fig_path, dpi=300)
            #plt.show()
            plt.close()

        #Save values min max and mean connectiivty across all blocks 
        df_con_values = pd.DataFrame(con_values, columns=["Block", "Min", "Max", "Mean", "SD"])
        con_values_path = os.path.join(save_dir, f"{base_filename}_iCoh_{ffmin}_{ffmax}_con_values_summary.csv")
        df_con_values.to_csv(con_values_path, index=False)

        print(f"✅ Saved stats for iCoh {ffmin}_{ffmax} all blocks to: {con_values_path}")
        print("\n📊 Summary of iCoh values for all blocks:")
        print(df_con_values.to_string(index=False))


# PLOT ALL THE BLOCKS TOGETHER 
fig, axes = plt.subplots(3, 3, figsize=(24, 24), constrained_layout=True)
fig.suptitle(f"Imaginary Coherence Connectivity Matrices by Block ({ffmin}-{ffmax} Hz)", fontsize=20)


vmin = min(np.min(mat) for mat in all_con)
vmax = max(np.max(mat) for mat in all_con)


ordered_channels = []
region_boundaries = []
channel_names = epochs.ch_names

for region, chs in regions.items():
    valid = [ch for ch in chs if ch in channel_names]
    ordered_channels.extend(valid)
    region_boundaries.append(len(ordered_channels))

        
# Loop through blocks and plot each
for idx, dense_con in enumerate(all_con):

    df = pd.DataFrame({
        "Ch1": [channel_names[i] for i in range(61) for j in range(61)],
        "Ch2": [channel_names[j] for i in range(61) for j in range(61)],
        "Value": dense_con.flatten()
        })

    pivot_df = df.pivot(index="Ch1", columns="Ch2", values="Value").fillna(0)


    pivot_ordered = pivot_df.loc[ordered_channels, ordered_channels]
    mask = pivot_ordered.values == 0

    row, col = divmod(idx, 3)
    ax = axes[row][col]

    sns.heatmap(
            pivot_ordered,
            cmap="coolwarm",
            center=0,
            square=True,
            mask=mask,
            xticklabels=False,
            yticklabels=False,
            cbar=False,
            ax=ax
        )
    ax.set_title(f"Block {idx}", fontsize=14)

    for boundary in region_boundaries[:-1]:
        ax.axhline(boundary, color='black', linewidth=1.0)
        ax.axvline(boundary, color='black', linewidth=1.0)


cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.95])
sns.heatmap(
    pivot_ordered,
    cmap="coolwarm",
    center=0,
    square=True,
    mask=mask,
    xticklabels=False,
    yticklabels=False,
    cbar_ax=cbar_ax,
    vmin=vmin,
    vmax=vmax
)
cbar_ax.set_ylabel("Imaginary Coherence", fontsize=14)


summary_fig_path = os.path.join(save_dir, f"{base_filename}_iCoh_{ffmin}_{ffmax}_all_blocks_grid.png")
plt.savefig(summary_fig_path, dpi=300)
plt.show()

print(f"✅ Multi-block heatmap saved to: {summary_fig_path}")
