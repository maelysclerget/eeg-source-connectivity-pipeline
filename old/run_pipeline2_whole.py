# run_pipeline2.py — Pipeline to compute the forward and the inverse solution for the subjects
# and extract the source time courses according to the specified atlas configuration

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

# INPUT PARAMETERS
subject = "TIP41"
session = 2
block = 2
whole_recording = False # set to True if you want to use the whole recording. If False, it will only consider
# the selected block and run the analysis on that.

# Choose a modality for source reconstruction
mode = "mixed" # "surface" or "volume" or "mixed"

# SUBJECTS_DIR SETUP
set_subjects_dir("/mnt/data/Semester_Projects/michelet/Source_Localization_Davide/Freesurfer/Subjects")
subjects_dir = get_subjects_dir()
check_freesurfer_subject(subject, subjects_dir)

# LOAD EEG DATA (personalized montage)
if whole_recording:
    #raise NotImplementedError("❌ whole recording loading not implemented yet.")
    raw_path = (
        "/mnt/data/Semester_Projects/michelet/Source_Localization_Davide/"
        f"OUT_montages_run_pipeline1/{subject}/ses_{session}/{subject}_ses{session}_whole_personalized_montage.fif"
    )
    # raw_path = (".fif") to the path where the whole recording (from the OUT montages dir)
else:
    raw_path = (
        "/mnt/data/Semester_Projects/michelet/"
        "Source_Localization_Davide/OUT_montages_run_pipeline1/"
        f"{subject}/ses_{session}/{subject}_ses{session}_block{block}_personalized_montage.fif"
    )
raw = mne.io.read_raw_fif(raw_path, preload=True)

# DEBUG to see it there are any channels with null location info
# In case they exist drop them because they will create issues in the inverse problem
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
    print("⚠️ Dropping EEG channels without valid locations (otherwise fails in the inverse problem):", bad_eeg)
    raw.drop_channels(bad_eeg)
#raise SystemExit("📝 Check EEG channel locations and dropped bad channels.")
#Checkpoint: raw data loaded with personalized montage and null channels removed ✅

# BEM
bem = create_bem_solution(subject, subjects_dir)
#raise SystemExit("📝 Check BEM creation.")
#Checkpoint: BEM model created correctly ✅

# SOURCE SPACE
if mode == "surface":
    src = create_surface_source_space(subject, subjects_dir)

elif mode == "volume":
    src = create_striatum_volume_source_space(subject, subjects_dir, bem)

elif mode == "mixed":
    src = create_mixed_source_space(subject, subjects_dir, bem)
    print(src)
    for s in src:
        print("TYPE:", s['type'], "NUSE:", s['nuse'], "SEG_NAME:", s.get('seg_name'))
    # raise SystemExit("📝 Check the mixed source space creation.")
    # checkpoint to see that mixed source space is created correctly ✅

else:
    raise ValueError("❌ mode must be 'surface', 'volume', or 'mixed'")

print("📝 CHECK THE NATURE OF THE SOURCE SPACE:")
print(src)
#raise SystemExit("Make sure that the source space is correctly created before proceeding.")
#Checkpoint: surface, volume, mixed they all work and are well created as source spaces ✅

# FORWARD MODEL: read the coregistration transformation matrix based on the previous coregistration step
trans_path = f"4)coreg_trans/{subject}_ses{session}_trans.fif"
trans = mne.read_trans(trans_path)

fwd = create_forward_solution(raw.info, trans, src, bem)
#raise SystemExit("📝 Forward solution creation.")
#Checkpoint: creation of the forward solution for all 3 modalities with free orientation ✅

# orientation handling (surface = fixed; volume/mixed = free) for visualization and leadfield analysis
fwd_oriented = orient_forward_solution(fwd, mode) #here we can include forward solution with fixed orientation only for surface
# in all the other cases, dipoles aren't oriented but kept in their 3 directions, in a free orientation configuration

leadfield = fwd_oriented["sol"]["data"]
# OUTPUT DIRECTORIES to save plots for the leadfield and sensitivity maps
out_dir = Path("OUT_source_estimates_run_pipeline2")

if whole_recording:
    #raise NotImplementedError("❌ whole recording output path not implemented yet.")
    plot_dir = out_dir / subject / f"ses{session}" / f"whole_recording" / mode / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    base_tag = f"{subject}_ses{session}_whole-recording_src-{mode}"
else:
    plot_dir = out_dir / subject / f"ses{session}" / f"block{block}" / mode / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    base_tag = f"{subject}_ses{session}_block{block}_src-{mode}"

# SAVE LEADFIELD MATRIX in chunks of 500 dipoles per block
# This way allows to have a leadfield matrix visualization for the huge number of diapoles splitted in multiple images
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
print(f"📂 Saved leadfield in {int(np.ceil(n_dipoles/block_size))} chunks.")

# EEG SENSITIVITY MAP (normalized dipoles column norms)
eeg_map = mne.sensitivity_map(fwd_oriented, ch_type="eeg", mode="fixed")
fig = plt.figure(figsize=(8, 4))
plt.hist(eeg_map.data.ravel(), bins=50, color="c")
plt.title(f"EEG Sensitivity — {mode}")
plt.tight_layout()
fig.savefig(plot_dir / f"{base_tag}_eeg_sensitivity.png", dpi=300)
plt.close(fig)
print(f"📂 Saved EEG dipole sensitivity map.")

# DIPOLES COLUMN NORMS (column norms of the leadfield matrix)
col_norms = np.linalg.norm(leadfield, axis=0)
fig = plt.figure(figsize=(8, 4))
plt.hist(col_norms, bins=40, color="purple")
plt.title("Dipole Norms Distribution")
plt.tight_layout()
fig.savefig(plot_dir / f"{base_tag}_dipole_norms.png", dpi=300)
plt.close(fig)
print(f"📂 Saved dipole norms distribution.")

# CHANNEL SENSITIVITY (row norms of the leadfield matrix)
row_norms = np.linalg.norm(leadfield, axis=1)
fig = plt.figure(figsize=(10, 4))
plt.bar(np.arange(len(row_norms)), row_norms, color="teal")
plt.title("Norm EEG Channel Sensitivity")
plt.tight_layout()
fig.savefig(plot_dir / f"{base_tag}_channel_sensitivity.png", dpi=300)
plt.close(fig)
print(f"📂 Saved channel norm sensitivity distribution.")

#raise SystemExit("📝 Check that the forward solution analysis is completed, orientations are good and plots are saved.")
# checkpoint: forward solution oriented correctly and leadfield plots saved for the 3 modalities ✅

# ⚠️ other things to possibly check (SANITY CHECKS): depth weighting, number of ill conditioned dipoles, rank of the leadfield matrix... ecc.
# such checke could be informative in terms of the source space definition and the associated forward solution quality

# NOISE COVARIANCE MATRIX: needs to be inlcuded from a baseline period because it accounts for those 
# components of the signal that can't be modelled via the forward solution, so to take them into account
# and avoid accentuating signals that should not be described because of being noise
print("🕒 Computing noise covariance matrix...")
noise_cov = mne.compute_raw_covariance(raw)
diag_cov = np.diag(noise_cov.data)
print("Mean noise variance:", diag_cov.mean())
print("Min / Max noise variance:", diag_cov.min(), diag_cov.max())
# Noise covaraince values should be low because otherwise the model doesn't well capture the real activity of the sources
print("✅ Noise covariance matrix computed.")
#raise SystemExit("📝 Ensure proper calculation of the noise covariance matrix.")
# checkpoint: compare values of the min / max noise covariances!!! ✅

# ⚠️ HERE WE MIGHT WANT TO COMPUTE THE NOISE COVARIANCE ON A BASELINE PERIOD
# INSTEAD OF USING THE SIGNALS FROM THE CONSIDERED BLOCK
# One may want to select the resting state periods before the task execution

# INVERSE OPERATOR
print("🕒 Creating inverse operator...")
inverse_operator = create_inverse_operator(
    raw.info,
    fwd, #use fwd (not fwd_oriented) and change loose/depth for surface
    noise_cov,
    mode
)
print("✅ Inverse operator created.")
# raise SystemExit("📝 Ensure proper calculation of the inverse operator.")
# CHECKPOINT: inverse operator was created successfully for all 3 modalities starting from the fwd and considering ad-hoc loose and depth✅

# COMPUTE SOURCE TIME COURSES (STC)
print("🕒 Computing source time courses (STC)...")
stc = compute_stc(
    raw,
    inverse_operator,
    lambda2=1.0/9, # because SNR = 3 according to MNE standard guidelines, and lambda2 = 1/SNR^2
    method="sLORETA" # Can be:
    # MNE = Minimum Norm Estimate, the basic inverse solution (time series of dipole amplitudes)
    # dSPM = dynamic Statistical Parametric Mapping, normalizes MNE by noise variance to produce Z-scores activation maps (answers to question: "is this dipole active compared to noise?")
	# sLORETA = standardized Low Resolution Electromagnetic Tomography, further normalizes dSPM for better localization and considers spatial smoothness.
    # Divides the time source estimates by an estimate of the standard deviation of the solution at each dipole location. Being the the variance bigger at deeper sources (due to uncertanty), this rescales the weight of deep sources that become comparable with surface ones.
)
print("✅ STC computed.")
# print("STC rows:", stc.data.shape[0])
# print("SRC total nuse:", sum(s['nuse'] for s in src))
# raise SystemExit("📝 Ensure proper computation of the STC.")
# CHECKPOINT: STC computed successfully for all 3 modalities ✅

# LABEL EXTRACTION TIME SERIES FOR THE 3 CASES: surface only estimation, volume only estimation, mixed surface + volume estimation
if whole_recording:
    out_dir_ts = out_dir / subject / f"ses{session}" / f"whole_recording" / mode
    out_dir_ts.mkdir(parents=True, exist_ok=True)
else:
    out_dir_ts = out_dir / subject / f"ses{session}" / f"block{block}" / mode
    out_dir_ts.mkdir(parents=True, exist_ok=True)

src_fwd = fwd['src'] # use the fwd obtained with the src_fwd creation function after fwd computation
# (some modifications to source spacecould be applied internally by MNE)

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