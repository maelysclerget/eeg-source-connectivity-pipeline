"""
source_utils1.py

Utilities for EEG source localization STEP 1 (before coregistration):

1. Personalization of EEG montage using neuronavigation coordinates.
2. Creation of scalp surfaces for high-resolution visualization.
3. Creation of BEM layers via the watershed algorithm.

Notes:
- Assumes FreeSurfer recon-all has already been run for each subject.
- These functions are thought to be called inside a higher-level pipeline script
  that loops over subjects / sessions / blocks.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import mne
import mne.bem
from mne.io.constants import FIFF
from matplotlib import pyplot as plt


# --- Logging utilities ---
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

def configure_basic_logging(log_path: str | Path | None = None,
                            level: int = logging.INFO) -> None:
    """
    Configure a simple logging setup for the pipeline.

    Parameters:
    log_path : str | Path | None
        If provided, logs will be written to this file in addition to stdout.
    level : int
        Logging level, e.g. logging.INFO or logging.DEBUG.
    """
    handlers = [logging.StreamHandler()]

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="a"))

    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        handlers=handlers,
    )

def set_subjects_dir(subjects_dir: str | Path) -> Path:
    """
    Convenience wrapper to set the SUBJECTS_DIR both as env var
    and as a resolved Path object.
    """
    subjects_dir = Path(subjects_dir).absolute()
    subjects_dir.mkdir(parents=True, exist_ok=True)
    os.environ["SUBJECTS_DIR"] = str(subjects_dir)
    logger.info(f"SUBJECTS_DIR set to: {subjects_dir}")
    return subjects_dir

# --- FreeSurfer / SUBJECTS_DIR helpers ---
def get_subjects_dir(subjects_dir: str | Path | None = None) -> Path:
    """
    Resolve the SUBJECTS_DIR used for FreeSurfer / MNE.

    Parameters: 
    subjects_dir : str | Path | None
        If provided, this path will be used.
        If None, the environment variable SUBJECTS_DIR must be set.

    Returns:
    Path
        Path object pointing to the SUBJECTS_DIR.

    Raises: 
    RuntimeError
        If SUBJECTS_DIR is not provided and not set in the environment.
    """
    if subjects_dir is not None:
        subjects_dir = Path(subjects_dir)
    else:
        env = os.getenv("SUBJECTS_DIR", None)
        if env is None:
            raise RuntimeError(
                "SUBJECTS_DIR is not set. Provide `subjects_dir` "
                "or set the environment variable SUBJECTS_DIR."
            )
        subjects_dir = Path(env)

    if not subjects_dir.exists():
        raise FileNotFoundError(f"SUBJECTS_DIR does not exist: {subjects_dir}")

    logger.info(f"Using SUBJECTS_DIR: {subjects_dir}")
    return subjects_dir


def check_freesurfer_subject(subject: str,
                             subjects_dir: str | Path | None = None) -> Path:
    """
    Check that the FreeSurfer subject directory exists.

    Parameters:
    subject : str
        Subject identifier.
    subjects_dir : str | Path | None
        FreeSurfer SUBJECTS_DIR base path.

    Returns:
    Path
        Path to the subject's FreeSurfer directory.

    Raises:
    FileNotFoundError
        If the subject directory does not exist.
    """
    subjects_dir = get_subjects_dir(subjects_dir)
    subject_path = subjects_dir / subject

    if not subject_path.exists():
        raise FileNotFoundError(
            f"Subject directory {subject_path} not found. "
            "Run recon-all first."
        )

    logger.info(f"Found FreeSurfer subject directory: {subject_path}")
    return subject_path


# Montage personalization
def _load_fiducials_from_xml(xml_path: str | Path) -> dict[str, np.ndarray]:
    """
    Load fiducial coordinates (nasion, lpa, rpa) from a BaseMarker XML file.

    Parameters:
    xml_path : str | Path
        Path to the XML file containing locator registration information.

    Returns:
    dict
        Dictionary with keys 'nasion', 'lpa', 'rpa' and 3D coordinates in meters.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"Fiducials XML not found: {xml_path}")

    tree = ET.parse(xml_path)
    root = tree.getroot()

    coordinates = []
    for marker in root.findall("Marker"):
        colvec = marker.find("ColVec3D")
        if colvec is not None:
            x = float(colvec.get("data0")) / 1000.0  # mm -> m
            y = float(colvec.get("data1")) / 1000.0
            z = float(colvec.get("data2")) / 1000.0
            coordinates.append([x, y, z])

    if len(coordinates) < 3:
        raise RuntimeError(
            f"Expected at least 3 fiducial markers in {xml_path}, "
            f"found {len(coordinates)}."
        )

    labels = ["nasion", "rpa", "lpa"]
    landmarks = dict(zip(labels, coordinates[:3]))

    for label, coord in landmarks.items():
        logger.info(f"Fiducial {label}: {coord}")

    return landmarks


def _load_neuronavigation_csv(csv_path: str | Path) -> pd.DataFrame:
    """
    Load neuronavigation coordinates from CSV.

    Expected columns: 'id', 'x', 'y', 'z' (in mm).

    Parameters:
    csv_path : str | Path
        Path to the neuronavigation CSV file.

    Returns:
    pd.DataFrame
        DataFrame with channel IDs and coordinates (still in mm).
    """
    csv_path = Path(csv_path)
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

    logger.info(
        f"Loaded neuronavigation data from {csv_path} with {len(df)} entries."
    )
    return df

# function for the personalization of the montage
def personalize_montage(
    fif_path: str | Path,
    neuronavigation_path: str | Path,
    fiducials_path: str | Path,
    output_dir: str | Path,
    subject: str,
    session: int,
    block: int,
    coord_frame: str = "head",
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
        Path to the BaseMarker XML file containing fiducial coordinates.
    output_dir : str | Path
        Directory where the personalized montage and plots will be saved.
    subject : str
        Subject identifier.
    session : int
        Session number.
    block : int
        Block number.
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
    output_dir = Path(output_dir/f"{subject}/ses_{session}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not fif_path.exists():
        raise FileNotFoundError(f"EEG .fif file not found: {fif_path}")

    logger.info(
        f"Loading raw FIF for subject {subject}, session {session}, "
        f"block {block} from {fif_path}"
    )
    raw = mne.io.read_raw_fif(
        fif_path,
        allow_maxshield=False,
        preload=True,
        on_split_missing="raise",
        verbose="ERROR",
    )

    # Channel / dig info for sanity check
    ch_names_raw = raw.info["ch_names"]
    dig_points_raw = raw.info.get("dig", [])
    logger.info(
        f"Raw has {len(ch_names_raw)} channels and {len(dig_points_raw)} "
        f"digitization points."
    )

    # Load neuronavigation coordinates
    neuronavigation_df = _load_neuronavigation_csv(neuronavigation_path)
    neuronavigation_channels = neuronavigation_df["id"].tolist()

    # Find intersection between amplifier channels and neuronavigation
    amplifier_channels = ch_names_raw
    in_both = sorted(set(neuronavigation_channels).intersection(amplifier_channels))

    if len(in_both) == 0:
        raise RuntimeError(
            f"No shared channels between amplifier and neuronavigation for "
            f"subject {subject}, session {session}."
        )

    logger.info(
        f"{len(in_both)} shared channels between amplifier and "
        f"neuronavigation for subject {subject}, session {session}."
    )

    # Channel positions (in meters)
    ch_pos_shared = {
        row["id"]: np.array(
            [row["x"] / 1000.0, row["y"] / 1000.0, row["z"] / 1000.0]
        )
        for _, row in neuronavigation_df.iterrows()
        if row["id"] in in_both
    }

    # Load fiducials
    fiducials = _load_fiducials_from_xml(fiducials_path)

    # Build montage
    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos_shared,
        nasion=fiducials["nasion"],
        lpa=fiducials["lpa"],
        rpa=fiducials["rpa"],
        coord_frame=coord_frame,
    )

    # Subset raw to shared channels only
    raw_subset = raw.copy().pick(picks=in_both)

    logger.info("Setting personalized montage on raw subset.")
    raw_subset.set_montage(montage, on_missing="warn")

    # Optional visualization
    if plot:
        fig = raw_subset.plot_sensors(kind="topomap", show_names=True)
        fig_path = (
            output_dir
            / f"{subject}_ses{session}_block{block}_personalized_montage.png"
        )
        plt.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved personalized montage plot to {fig_path}")

    ###ADD THIS BEFORE SAVING TO ENSURE CHANNELS HAVE POSITION INFO###
    bad_eeg = []

    for ch in raw_subset.info['chs']:
        if ch['kind'] == FIFF.FIFFV_EEG_CH:
            loc = np.array(ch['loc'][:3])
            if np.allclose(loc, 0) or np.isnan(loc).any():
                bad_eeg.append(ch['ch_name'])
    if bad_eeg:
        logger.warning(f"Dropping EEG channels with missing location: {bad_eeg}")
        raw_subset.drop_channels(bad_eeg)
    ######

    # Save fif with personalized montage
    if block == 0:
        save_path = output_dir / f"{subject}_ses{session}_whole_personalized_montage.fif"
        raw_subset.save(save_path, overwrite=True)
    else:
        save_path = output_dir / f"{subject}_ses{session}_block{block}_personalized_montage.fif"
        raw_subset.save(save_path, overwrite=True)
    logger.info(f"Saved personalized montage FIF to {save_path}")

    return raw_subset


# Scalp surfaces
def scalp_surfaces_exist(subject: str, subjects_dir: str | Path) -> bool:
    """
    Check whether scalp surfaces already exist for a given subject.
    If they exist, we can skip make_scalp_surfaces().

    Checks for the presence of at least one typical scalp surface file:
        - <subject>-head.fif

    Returns:
    bool
        True if scalp surfaces exist.
    """
    subj_bem = Path(subjects_dir) / subject / "bem"
    # Common scalp surface output created by mne.make_scalp_surfaces
    head_file = subj_bem / f"{subject}-head.fif"

    return head_file.exists()

def make_scalp_surfaces(
    subject: str,
    subjects_dir: str | Path | None = None,
    overwrite: bool = False,
    no_decimate: bool = True,
) -> list[Path]:
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

    subjects_dir = get_subjects_dir(subjects_dir)
    subj_path = subjects_dir / subject
    bem_dir = subj_path / "bem"
    bem_dir.mkdir(parents=True, exist_ok=True)

    # Skip if already exists
    if not overwrite and scalp_surfaces_exist(subject, subjects_dir):
        logger.info(f"Scalp surfaces already exist for {subject}. Skipping.")
        return []

    logger.info(
        f"Creating scalp surfaces for {subject} (overwrite={overwrite}, "
        f"no_decimate={no_decimate})."
    )

    surfaces = mne.bem.make_scalp_surfaces(
        subject=subject,
        subjects_dir=str(subjects_dir),
        force=overwrite,
        no_decimate=no_decimate,
        overwrite=overwrite,
    )

    out_paths = []
    for s in surfaces:
        surf_path = bem_dir / f"{subject}_{s['id']}_scalp.surf"
        out_paths.append(surf_path)

    logger.info(
        f"Scalp surface generation completed for {subject}. "
        f"Generated {len(out_paths)} surfaces."
    )

    return out_paths


# BEM via watershed
def watershed_exists(subject: str, subjects_dir: str | Path) -> bool:
    """
    Return True if the watershed directory exists and is non-empty.

    This is the safest and most robust check:
    - MNE, FreeSurfer legacy and other tools may produce different file names.
    - If the directory is non-empty, we assume watershed has been run.
    - Only re-run if overwrite=True.
    """
    watershed_dir = Path(subjects_dir) / subject / "bem" / "watershed"

    if not watershed_dir.exists():
        return False

    # If the directory contains ANY file → watershed already done
    if any(watershed_dir.iterdir()):
        return True

    return False

def make_bem_watershed(
    subject: str,
    subjects_dir: str | Path | None = None,
    overwrite: bool = False,
    show: bool = False,
) -> Path:
    """
    Create BEM layers using the watershed algorithm.

    SKIPS generation if watershed directory already exists and is non-empty,
    unless overwrite=True.

    Uses robust detection of watershed presence.
    """

    subjects_dir = get_subjects_dir(subjects_dir)
    subj_path = subjects_dir / subject
    watershed_dir = subj_path / "bem" / "watershed"

    # SKIP if directory exists AND is non-empty AND overwrite=False
    if not overwrite and watershed_exists(subject, subjects_dir):
        logger.info(f"Watershed already exists for {subject}. Skipping.")
        return watershed_dir
    
    # Otherwise, compute watershed BEM
    logger.info(
        f"Running watershed BEM for {subject} "
        f"(overwrite={overwrite}, show={show})."
    )

    mne.bem.make_watershed_bem(
        subject=subject,
        subjects_dir=str(subjects_dir),
        overwrite=overwrite,
        show=show,
    )

    logger.info(
        f"Watershed BEM completed successfully for subject {subject}. "
        f"Files located in: {watershed_dir}"
    )

    return watershed_dir