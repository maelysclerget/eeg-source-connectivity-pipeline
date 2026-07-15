#Author Stavriani Skarvelaki
"""
Script for EEG source localization STEP 1 (before coregistration):

1. DIGITALIZATION of EEG montage using neuronavigation coordinates.
2. Creation of scalp surfaces for high-resolution visualization.
3. Creation of BEM layers via the watershed algorithm.

Notes:
- Assumes FreeSurfer recon-all has already been run for each subject.

"""
import os
import re
import glob
import mne
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from mne.io.constants import FIFF

BIDS_ROOT = "/work/uphummel/studies/tTIS-EEG/data/raw/EEG"
DERIVATIVES_DIR = "/work/uphummel/studies/tTIS-EEG/derivatives/EEG"    
FREESURFER_DIR = "/work/uphummel/studies/tTIS-EEG/derivatives/MRI/freesurfer"  
NEURONAVIGATION_DIR = "/work/uphummel/studies/tTIS-EEG/data/source/Neuronavigation"
MRI_DIR = "/work/uphummel/studies/tTIS-EEG/data/raw/MRI"

def load_neuronavigation_csv(subject, ses) -> pd.DataFrame:
    """
    Load neuronavigation coordinates from CSV. Be careful with subject and session formatting, as these files do not 
    comply with BIDS. Expected columns: 'id', 'x', 'y', 'z' (in mm). Important to have already converted the neuronavigation files with the juppyter 
    notebook script by SS. 

    Parameters:
    subject : str
        Subject identifier.
    session : int
        Session number.

    Returns:
    pd.DataFrame
        DataFrame with channel IDs and coordinates (still in mm).
    """
    csv_path = Path(NEURONAVIGATION_DIR) / f"{subject}/ses_{ses}/{subject}_ses{ses}_EEG_montage.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Neuronavigation CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.dropna(how="any")
    df = df.drop_duplicates(subset=["id"], keep="first")

    required_cols = {"id", "x", "y", "z"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Neuronavigation CSV missing required columns: {required_cols}"
        )

    print(f"Loaded neuronavigation data from {csv_path} with {len(df)} entries.")
    return df

def load_fiducials_from_csv(subject, ses) -> pd.DataFrame:
    """
    Load neuronavigation coordinates for fiducials from CSV. Be careful with subject and session formatting, as these files do not 
    comply with BIDS. Expected columns: 'id', 'x', 'y', 'z' (in mm). Important to have already converted the neuronavigation files with the jupyter 
    notebook script by SS. 

    Parameters:
    subject : str
        Subject identifier.
    session : int
        Session number.

    Returns:
    pd.DataFrame
        DataFrame with channel IDs and coordinates (still in mm).
    """
    csv_path = Path(NEURONAVIGATION_DIR) / f"{subject}/ses_{ses}/{subject}_ses{ses}_fiducials.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Neuronavigation CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df.dropna(how="any")
    df = df.drop_duplicates(subset=["id"], keep="first")

    required_cols = {"id", "x", "y", "z"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Neuronavigation CSV missing required columns: {required_cols}")

    fiducials = {
        row["id"]: np.array(
            [row["x"], row["y"], row["z"]],
            dtype=float
        ) / 1000.0
        for _, row in df.iterrows()
    }

    print(f"Loaded neuronavigation fiducials from {csv_path} with {len(df)} entries.")
    
    return fiducials

# function for the personalization of the montage
def personalize_montage(
    fif_path: str | Path,
    subject: str,
    ses: int,
    coord_frame: str = "ras",
    plot: bool = True,
) -> mne.io.BaseRaw:
    
    """
    Personalize the EEG montage using neuronavigation coordinates.

    Parameters:
    fif_path : str | Path
        Path to the pre-processed EEG .fif file.
    neuronavigation_path : str | Path
        Path to the neuronavigation CSV file containing electrode positions.
    fiducials_path : str | Path
        Path to the fiducials .csv file containing fiducial coordinates.
    output_dir : str | Path
        Directory where the personalized montage and plots will be saved.
    subject : str
        Subject identifier.
    session : int
        Session number.
    coord_frame : {'head', 'ras'}, default 'head'
        Coordinate frame of the neuronavigation data. MNE typically expects
        'head'. Use 'ras' only if you know coordinates are in MRI RAS frame
        and have the appropriate transforms.
    plot : bool, default True
        Whether to generate and save a sensor topomap.

    Returns:
    raw_subset : mne.io.Raw
        Raw object with personalized montage for the channels present in both
        the amplifier and neuronavigation.
    """
    fif_path = Path(fif_path)
    output_folder = Path(DERIVATIVES_DIR)/f"sub-{subject}/ses-{ses}/source_reconstruction/personalized_montage"
    output_folder.mkdir(parents=True, exist_ok=True)
    #os.chmod(output_folder, 0o2770)

    if not fif_path.exists():
        raise FileNotFoundError(f"EEG .fif file not found: {fif_path}")

    print(f"Loading raw FIF for subject {subject}, session {ses}")
    raw = mne.io.read_raw_fif(fif_path, preload=True, verbose="ERROR")

    # Channel / dig info for sanity check
    ch_names_raw = raw.info["ch_names"]
    dig_points_raw = raw.info.get("dig", [])
    print(f"Raw has {len(ch_names_raw)} channels and {len(dig_points_raw)} digitization points.")


    # Find intersection between amplifier channels and neuronavigation
    # Use uppercase only for matching
    neuronavigation_df = load_neuronavigation_csv(subject, ses)
    amplifier_channels = ch_names_raw

    raw_map = {ch.upper(): ch for ch in amplifier_channels} #creates a dictionnary where the key is the uppercase version, and the value is the real raw channel name.
    neuronavigation_df["id_norm"] = neuronavigation_df["id"].astype(str).str.upper() #creates a normalized version of the id column for matching

    in_both_norm = sorted(set(neuronavigation_df["id_norm"]).intersection(raw_map.keys()))
    in_both = [raw_map[ch] for ch in in_both_norm]

    if len(in_both) == 0:
        raise RuntimeError(f"No shared channels between amplifier and neuronavigation for subject {subject}, session {ses}.")

    print(
        f"{len(in_both)} shared channels between amplifier and neuronavigation "
        f"after normalizing channel labels for subject {subject}, session {ses}."
    )

    nav_only = sorted(set(neuronavigation_df["id_norm"]) - set(raw_map.keys()))
    if nav_only:
        print(f"Entries in neuronavigation CSV but not in raw FIF: {nav_only}")

    # Channel positions (in meters)
    # Use raw channel names as keys, so MNE can match the montage to raw.ch_names
    ch_pos_shared = {
        raw_map[row["id_norm"]]: np.array(
            [row["x"] / 1000.0, row["y"] / 1000.0, row["z"] / 1000.0]
        )
        for _, row in neuronavigation_df.iterrows()
        if row["id_norm"] in in_both_norm
    } 
    
    # PREVIOUS VERSION
    """     # Load neuronavigation coordinates
   neuronavigation_df = load_neuronavigation_csv(subject, ses)
   neuronavigation_channels = neuronavigation_df["id"].tolist()


   # Find intersection between amplifier channels and neuronavigation
   amplifier_channels = ch_names_raw
   in_both = sorted(set(neuronavigation_channels).intersection(amplifier_channels))


   if len(in_both) == 0:
       raise RuntimeError(f"No shared channels between amplifier and neuronavigation for subject {subject}, session {ses}.")
   print(f"{len(in_both)} shared channels between amplifier and neuronavigation for subject {subject}, session {ses}.")


   # Channel positions (in meters)
   ch_pos_shared = {
       row["id"]: np.array(
           [row["x"] / 1000.0, row["y"] / 1000.0, row["z"] / 1000.0]
       )
       for _, row in neuronavigation_df.iterrows()
       if row["id"] in in_both
   } """

    # Load fiducials
    fiducials = load_fiducials_from_csv(subject, ses)

    extra_fiducials = ["left exocantion", "right exocantion", "inion",] 
    ## Optional extra anatomical points from neuronavigation.
    # Note: MNE's hsp parameter is intended for head-shape/scalp points, not fiducials.
    # These points are not replaced by FreeSurfer points; they are stored as extra
    # digitization points and may help visual inspection/coregistration only if they
    # are in the same coordinate frame as the electrodes and fiducials.
    hsp = np.array([fiducials[name] for name in extra_fiducials if name in fiducials])

    # Build montage
    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos_shared,
        nasion=fiducials["nasion"],
        lpa=fiducials["left pre auricular point"],
        rpa=fiducials["right pre auricular point"],
        hsp=hsp,
        coord_frame=coord_frame
    ) 

    # Subset raw to shared channels only
    raw_subset = raw.copy().pick(picks=in_both)

    print("Setting personalized montage on raw subset.")
    raw_subset.set_montage(montage, on_missing="warn")
    print(f"After setting montage: {len(raw_subset.info.get('dig', []))} digitization points.")

    # Optional visualization
    if plot:
        fig = raw_subset.plot_sensors(kind="topomap", show_names=True)
        fig_path = (output_folder/ f"{subject}_ses{ses}_personalized_montage.png")
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved personalized montage plot to {fig_path}")

    ###ADD THIS BEFORE SAVING TO ENSURE CHANNELS HAVE POSITION INFO###
    # Remove EEG channels that were not assigned valid 3D coordinates.
    # This is independent of raw.info["bads"].
    bad_eeg = []

    for ch in raw_subset.info['chs']:
        if ch['kind'] == FIFF.FIFFV_EEG_CH:
            loc = np.array(ch['loc'][:3])
            if np.allclose(loc, 0) or np.isnan(loc).any():
                bad_eeg.append(ch['ch_name'])
    if bad_eeg:
        print(
        f"The following EEG channels have no valid position and will be removed: "
        f"{bad_eeg}"
    )
        raw_subset.drop_channels(bad_eeg) #does it take th bad annotated channels or the ones with missing location? 
    ######

    # Save fif with personalized montage
    save_path = output_folder/ f"{subject}_ses{ses}_personalized_montage.fif"
    raw_subset.save(save_path, overwrite=True)
    print(f"Saved personalized montage FIF to {save_path}")
    
    return raw_subset

def make_scalp_surfaces(subject: str, overwrite: bool = False, no_decimate: bool = False) -> list[Path]:
    
    """
    Create scalp surfaces for high-resolution visualization in FreeSurfer.

    This version automatically SKIPS computation if scalp surfaces already exist, unless overwrite=True.

    Parameters:
    subject : str
        Subject identifier.
    subjects_dir : str | Path | None
        FreeSurfer SUBJECTS_DIR base path.
    overwrite : bool
        If True, recompute even if surfaces exist.
    no_decimate : bool
        If True, produce high-resolution surfaces.

    Returns:
    list of Path
        Paths to the generated scalp surface files.
    """
    bem_dir = Path(FREESURFER_DIR)/f"sub-{subject}_ses-baseline/bem"

    if not bem_dir.exists():
        bem_dir.mkdir(parents=True, exist_ok=True)
        
    # Skip only when all scalp surface resolutions already exist.
    required_head_surfaces = [
        bem_dir / f"sub-{subject}_ses-baseline-head-dense.fif",
        bem_dir / f"sub-{subject}_ses-baseline-head-medium.fif",
        bem_dir / f"sub-{subject}_ses-baseline-head-sparse.fif",
    ]

    if not overwrite and all(p.exists() for p in required_head_surfaces):
        print(f"Scalp surfaces already exist for {subject}. Skipping.")
        return required_head_surfaces

    print(
        f"Creating scalp surfaces for {subject} (overwrite={overwrite}, "
        f"no_decimate={no_decimate})."
    )

    surfaces = mne.bem.make_scalp_surfaces(
        subject=f"sub-{subject}_ses-baseline",
        subjects_dir=Path(FREESURFER_DIR),
        force=overwrite,
        mri="T1.mgz",
        no_decimate=no_decimate,
        overwrite=overwrite,
    )

    """out_paths = []
    for s in surfaces:
        surf_path = bem_dir / f"{subject}_{s['id']}_scalp.surf"
        out_paths.append(surf_path) """
        
    out_paths = list(bem_dir.glob("*head-*.fif")) # look into the folder after creating the files 

    print(
        f"Scalp surface generation completed for {subject}. "
        f"Generated {len(out_paths)} surfaces."
    )

    return out_paths

def make_bem_watershed(
    subject: str,
    overwrite: bool = False,
    show: bool = False,
) -> Path:
    """
    Create BEM layers using the watershed algorithm.

    SKIPS generation if watershed directory already exists and is non-empty,
    unless overwrite=True.

    Uses robust detection of watershed presence.
    """
    output_folder = Path(FREESURFER_DIR)/f"sub-{subject}_ses-baseline/bem"
    watershed_dir = output_folder / "watershed"

    output_folder.mkdir(parents=True, exist_ok=True)
    
    
    required_surfaces = [
        output_folder / "inner_skull.surf",
        output_folder / "outer_skull.surf",
        output_folder / "outer_skin.surf",
    ]

    if not overwrite and all(p.exists() for p in required_surfaces):
        print(f"Watershed BEM already exists for {subject}. Skipping BEM creation.")
    else:
        print(
            f"Running watershed BEM for {subject} "
            f"(overwrite={overwrite}, show={show})."
        )

        volume = Path(FREESURFER_DIR) / f"sub-{subject}_ses-baseline" / "mri" / "T1.mgz"
    
        mne.bem.make_watershed_bem(
            subject=f"sub-{subject}_ses-baseline",
            subjects_dir=FREESURFER_DIR,
            volume=str(volume),
            overwrite=overwrite,
            show=show,
        )

        print(
            f"Watershed BEM completed successfully for subject {subject}. "
            f"Files located in: {watershed_dir}"
        )

    fig = mne.viz.plot_bem(
        subject=f"sub-{subject}_ses-baseline",
        subjects_dir=FREESURFER_DIR,
        orientation="coronal",
        show=False,
    )

    fig_path = output_folder / f"{subject}_bem.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved BEM visualization to {fig_path}")

    return watershed_dir

def process_file(input_path):
    input_file = Path(input_path)   

    print(f"Processing: {input_file}")

    try:

        personalize_montage(fif_path=input_file, subject=subject, ses=session)

        if str(session) == "1":
            make_scalp_surfaces(subject)
            make_bem_watershed(subject)
        else:
            print(
                f"Skipping BEM creation for sub-{subject} ses-{session}; "
                "BEM is shared across sessions and is created from ses-1."
            )

    except Exception as err:
        print(f"Error processing file {input_file}")
        print(err)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Prepare personalized EEG montage and BEM surfaces"
    )

    parser.add_argument(
        "--subject",
        required=True,
        help="Subject ID (e.g. 41Y01)"
    )

    parser.add_argument(
        "--session",
        required=True,
        help="Session ID (e.g. 1)"
    )

    parser.add_argument(
        "--task",
        required=True,
        help="Task folder name (e.g. task, RSpre, RSpost)"
    )

    args = parser.parse_args()

    print(
        f"DEBUG subject={args.subject!r}, "
        f"session={args.session!r}, "
        f"task={args.task!r}"
    )

    subject = args.subject
    session = args.session
    task = args.task

    # Folder containing the representative FIF
    input_dir = (
        Path(DERIVATIVES_DIR)
        / f"sub-{subject}"
        / f"ses-{session}"
        / task
    )

    if not input_dir.exists():
        raise RuntimeError(f"Directory not found: {input_dir}")

    # Representative preprocessed FIF
    fif_path = sorted(input_dir.glob(f"sub-{subject}_ses-{session}_task-{task}_eeg_final_preprocessed_raw_source_level.fif"))

    print(
        f"Found {len(fif_path)} EEG files "
        f"for sub-{subject} ses-{session}"
    )

    if len(fif_path) == 0:
        raise RuntimeError(
            f"No representative FIF file found in {input_dir}"
        )

    representative_fif = fif_path[0]

    print(f"Using representative FIF: {representative_fif}")

    process_file(representative_fif)
