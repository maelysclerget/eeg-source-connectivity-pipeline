# NOTE: same pipeline as run_pipeline2_whole.py, with the only difference that here you can run across multiple patients,
# sessions, and blocks by specifying them in lists at the beginning of the script. Still, can decide what type of 
# source space to use (surface, volume, mixed) and whether to process whole recordings or single blocks.

# run_pipeline2_all.py — Pipeline to compute the forward and the inverse solution for the subjects
# and extract the source time courses according to the specified atlas configuration

##########################################################################################################
# SUGGESTION: for future coherence measurement and connectivity analysis, also consider the baseline by
# inlcuding the resting state data as block 0. This would make it easier the reconstruction in the source space
# for the baseline, just by integrating block ZERO in the current pipeline!!!
###########################################################################################################

"""
run_pipeline2.py

Pipeline for:
- building surface / volume (striatal) / mixed source spaces
- computing forward & inverse solutions
- extracting STC, cortical label time series, and striatal TS
- saving diagnostic plots (leadfield, sensitivity maps)

Modes:
    - "surface" → cortical surface (oct6)
    - "volume"  → striatum only (Caudate, Putamen, Accumbens)
    - "mixed"   → surface + striatum

Orientation:
    - surface → dipoles constrained normal to cortex (fixed)
    - volume/mixed → free orientation

Emojis legend for log messages:
- 📝 Debugging
- ⚠️ Warnings
- ✅ Success
- 🕒 Loading
- 📂 Saved
- ❌ Failure
"""

import mne
import numpy as np
import matplotlib
matplotlib.use("Agg") # used to block the backend display from plotting during the run
import matplotlib.pyplot as plt
from pathlib import Path
from mne.io.constants import FIFF

from source_utils1 import (
    configure_basic_logging,
    set_subjects_dir,
    get_subjects_dir,
    check_freesurfer_subject,
)

from source_utils2 import (
    create_surface_source_space,
    create_striatum_volume_source_space,
    create_mixed_source_space,
    create_bem_solution,
    create_forward_solution,
    orient_forward_solution,
    create_inverse_operator,
    compute_stc,
    extract_labels_all_modes,
    save_label_ts_as_fif,
)

# LOGGING
configure_basic_logging("logs2/pipeline.log")

# Same code as before, but here can list multiple subjects, sessions, blocks
subjects = ["TIP39"] #"TIP41", "TIP43"
sessions = [1] # [1, 2, 3]
blocks   = [1, 2, 3, 4, 5, 6, 7, 8, 9]

whole_recording = True # when True, processes whole recording
mode = "surface" # "surface", "volume", or "mixed"

# SUBJECTS_DIR SETUP
set_subjects_dir("/mnt/data/Semester_Projects/michelet/Source_Localization_Davide/Freesurfer/Subjects")
subjects_dir = get_subjects_dir()

# 🔵 blue point marks the start of the triple loop (subject, session, block)
for subject in subjects:
    for session in sessions:
        if whole_recording:
            blocks = [1] # because otherwise repeats the loop for all block types but processes every time the whole recording for the corresponding session
        for block in blocks:
            if whole_recording:
                print("\n\n" + "="*90)
                print(f"🔵 Running pipeline for SUBJECT={subject}  SESSION={session}  WHOLE RECORDING")
                print("="*90)
            else:
                print("\n\n" + "="*90)
                print(f"🔵 Running pipeline for SUBJECT={subject}  SESSION={session}  BLOCK={block}")
                print("="*90)

            # Make sure subject exists in Freesurfer
            try:
                check_freesurfer_subject(subject, subjects_dir)
            except Exception as e:
                print(f"❌ Subject {subject} missing in FreeSurfer → SKIP")
                continue

            # LOAD EEG DATA (personalized montage)
            if whole_recording:
                print("Whole recording mode on: skipping block wise processing but considering whole recording")
                raw_path = (
                    "/mnt/data/Semester_Projects/michelet/Source_Localization_Davide/"
                    f"OUT_montages_run_pipeline1/{subject}/ses_{session}/"
                    f"{subject}_ses{session}_whole_personalized_montage.fif"
                )
            else:
                raw_path = (
                    "/mnt/data/Semester_Projects/michelet/"
                    "Source_Localization_Davide/OUT_montages_run_pipeline1/"
                    f"{subject}/ses_{session}/{subject}_ses{session}_block{block}_personalized_montage.fif"
                )

            if not Path(raw_path).exists():
                print(f"⚠️ Raw file missing → SKIP {raw_path}")
                continue

            raw = mne.io.read_raw_fif(raw_path, preload=True) # very expensive operation
            # loads the whole recording into raw memory

            # DEBUG to see if there are channels with null location info
            print("📝 EEG channel locations from personalized montage")
            bad_eeg = []
            for ch in raw.info['chs']:
                if ch['kind'] == FIFF.FIFFV_EEG_CH:
                    name = ch['ch_name']
                    loc = np.array(ch['loc'][:3])
                    print(f"{name:>6} -> {loc}")
                    if np.allclose(loc, 0) or np.isnan(loc).any():
                        bad_eeg.append(name)

            print("⚠️ EEG channels with invalid locations:", bad_eeg)
            if bad_eeg:
                print("⚠️ Dropping channels:", bad_eeg)
                raw.drop_channels(bad_eeg)

            # ALL THE REST OF YOUR PIPELINE GOES HERE EXACTLY AS IT WAS
            # BEM
            bem = create_bem_solution(subject, subjects_dir)

            # SOURCE SPACE
            if mode == "surface":
                src = create_surface_source_space(subject, subjects_dir)
            elif mode == "volume":
                src = create_striatum_volume_source_space(subject, subjects_dir, bem)
            elif mode == "mixed":
                src = create_mixed_source_space(subject, subjects_dir, bem)
            else:
                raise ValueError("❌ mode must be 'surface', 'volume', or 'mixed'")

            print("📝 CHECK THE NATURE OF THE SOURCE SPACE:")
            print(src)

            # FORWARD MODEL
            trans_path = f"4)coreg_trans/{subject}_ses{session}_trans.fif"
            if not Path(trans_path).exists():
                print(f"❌ Missing TRANS file → SKIP {trans_path}")
                continue

            trans = mne.read_trans(trans_path)
            fwd = create_forward_solution(raw.info, trans, src, bem)
            fwd_oriented = orient_forward_solution(fwd, mode)

            leadfield = fwd_oriented["sol"]["data"]
            out_dir = Path("OUT_source_estimates_run_pipeline2")

            if whole_recording:
                plot_dir = out_dir / subject / f"ses{session}" / "whole_recording" / mode / "plots"
                base_tag = f"{subject}_ses{session}_whole-recording_src-{mode}"
            else:
                plot_dir = out_dir / subject / f"ses{session}" / f"block{block}" / mode / "plots"
                base_tag = f"{subject}_ses{session}_block{block}_src-{mode}"

            plot_dir.mkdir(parents=True, exist_ok=True) # make sure to create the plot directory also for the whole recording case

            print("🕒 Saving leadfield matrix plots...")
            block_size = 500
            n_dipoles = leadfield.shape[1]

            for start in range(0, n_dipoles, block_size):
                end = min(start + block_size, n_dipoles)
                fig = plt.figure(figsize=(10, 5))
                plt.imshow(leadfield[:, start:end], aspect="auto", cmap="RdBu_r")
                plt.title(f"Leadfield — dipoles {start} to {end - 1} — Mode: {mode}")
                plt.xlabel("Dipole index")
                plt.ylabel("EEG channel")
                plt.colorbar()
                plt.tight_layout()
                fig.savefig(plot_dir / f"{base_tag}_leadfield_{start}_{end-1}.png", dpi=300)
                plt.close(fig)

            # EEG SENSITIVITY MAP
            eeg_map = mne.sensitivity_map(fwd_oriented, ch_type="eeg", mode="fixed")
            fig = plt.figure(figsize=(8, 4))
            plt.hist(eeg_map.data.ravel(), bins=50, color="c")
            plt.title(f"EEG Sensitivity — {mode}")
            plt.tight_layout()
            fig.savefig(plot_dir / f"{base_tag}_eeg_sensitivity.png", dpi=300)
            plt.close(fig)

            # DIPOLE NORMS
            col_norms = np.linalg.norm(leadfield, axis=0)
            fig = plt.figure(figsize=(8, 4))
            plt.hist(col_norms, bins=40, color="purple")
            plt.title("Dipole Norms Distribution")
            fig.savefig(plot_dir / f"{base_tag}_dipole_norms.png", dpi=300)
            plt.close(fig)

            # CHANNEL SENSITIVITY
            row_norms = np.linalg.norm(leadfield, axis=1)
            fig = plt.figure(figsize=(10, 4))
            plt.bar(np.arange(len(row_norms)), row_norms, color="teal")
            plt.title("Norm EEG Channel Sensitivity")
            fig.savefig(plot_dir / f"{base_tag}_channel_sensitivity.png", dpi=300)
            plt.close(fig)

            # NOISE COVARIANCE
            print("🕒 Computing noise covariance matrix...")
            noise_cov = mne.compute_raw_covariance(raw)

            # INVERSE OPERATOR
            print("🕒 Creating inverse operator...")
            inverse_operator = create_inverse_operator(raw.info, fwd, noise_cov, mode)

            # STC
            print("🕒 Computing source time courses (STC)...")
            stc = compute_stc(raw, inverse_operator, lambda2=1.0/9, method="sLORETA")

            # LABEL TIME SERIES
            if whole_recording:
                out_dir_ts = out_dir / subject / f"ses{session}" / "whole_recording" / mode
            else:
                out_dir_ts = out_dir / subject / f"ses{session}" / f"block{block}" / mode

            out_dir_ts.mkdir(parents=True, exist_ok=True)
            src_fwd = fwd['src']

            print("🕒 Extracting label time series...")

            if mode == "surface":
                labels, label_ts, kind = extract_labels_all_modes(
                    stc, src_fwd, mode, subject, subjects_dir
                )
                roi_names = [lab.name for lab in labels]
                fname = out_dir_ts / f"{base_tag}_labels.fif"
                save_label_ts_as_fif(label_ts, roi_names, stc, fname)

            elif mode == "volume":
                roi_names, roi_ts, kind = extract_labels_all_modes(
                    stc, src_fwd, mode, subject, subjects_dir
                )
                fname = out_dir_ts / f"{base_tag}_labels.fif"
                save_label_ts_as_fif(roi_ts, roi_names, stc, fname)

            elif mode == "mixed":
                roi_dict, _, kind = extract_labels_all_modes(
                    stc, src_fwd, mode, subject, subjects_dir
                )

                fname_surf = out_dir_ts / f"{base_tag}_surface_labels.fif"
                save_label_ts_as_fif(
                    roi_dict["surface_ts"],
                    [lab.name for lab in roi_dict["surface_labels"]],
                    stc,
                    fname_surf,
                )

                fname_deep = out_dir_ts / f"{base_tag}_volume_labels.fif"
                save_label_ts_as_fif(
                    roi_dict["volume_ts"],
                    roi_dict["volume_labels"],
                    stc,
                    fname_deep,
                )

                print(f"📂 Saved MIXED (surface + deep) ROI time series in: {out_dir_ts}")

            if whole_recording:
                print(f"✅ FINISHED subject={subject}  session={session}  whole_recording")
            else:
                print(f"✅ FINISHED subject={subject}  session={session}  block={block}")