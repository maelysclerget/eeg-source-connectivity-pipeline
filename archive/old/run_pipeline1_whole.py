#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EEG-MRI Source Localization Pipeline
Fixed, clean, robust, and optimized version.
Change the flag 'whole_recording' to switch between the block-based processing mode and the whole-recording processing mode.
"""

import matplotlib
matplotlib.use("Agg")

import logging
import mne
import re
from pathlib import Path
import matplotlib.pyplot as plt

from source_utils1 import (
    configure_basic_logging,
    personalize_montage,
    make_scalp_surfaces,
    make_bem_watershed,
)

# LOGGING
configure_basic_logging("logs/pipeline.log")
logger = logging.getLogger(__name__)

# USER CONFIGURATION
subject_config = {
    "TIP39": {
        "sessions": [1, 2, 3],
        "basemarker": {
            1: Path("/home/skarvela/mnt/hummel-data/TI/TI_EEG/EEG/"
                    "Neuronavigation/subject/TIP39/ses_1/Session_20250528093942183/"
                    "Registrations/LocatorRegistrations/BaseMarker20250528111557686.xml"),
            2: Path("/home/skarvela/mnt/hummel-data/TI/TI_EEG/EEG/"
                    "Neuronavigation/subject/TIP39/ses_2/Session_20250604100610607/"
                    "Registrations/LocatorRegistrations/BaseMarker20250604111829848.xml")
        },
    },

    "TIP40": {
        "sessions": [1, 2, 3],
        "basemarker": {
            1: Path("/home/skarvela/mnt/hummel-data/TI/TI_EEG/EEG/"
                    "Neuronavigation/subject/TIP40/ses_1/Session_20250603092256669/"
                    "Registrations/LocatorRegistrations/BaseMarker20250603111518873.xml"),
            2: Path("/home/skarvela/mnt/hummel-data/TI/TI_EEG/EEG/"
                    "Neuronavigation/subject/TIP40/ses_2/Session_20250611135714783/"
                    "Registrations/LocatorRegistrations/BaseMarker20250611140239556.xml"),
        },
    },

    "TIP41": {
        "sessions": [1, 2, 3],
        "basemarker": {
            1: Path("/home/skarvela/mnt/hummel-data/TI/TI_EEG/EEG/"
                    "Neuronavigation/subject/TIP41/ses_1/Session_20250606092918255/"
                    "Registrations/LocatorRegistrations/BaseMarker20250606120700127.xml"),
            2: Path("/home/skarvela/mnt/hummel-data/TI/TI_EEG/EEG/"
                    "Neuronavigation/subject/TIP41/ses_2/Session_20250613100903523/"
                    "Registrations/LocatorRegistrations/BaseMarker20250613120520064.xml"),
        },
    },

    "TIP43": {
        "sessions": [1, 2, 3],
        "basemarker": {
            1: Path("/home/skarvela/mnt/hummel-data/TI/TI_EEG/EEG/"
                    "Neuronavigation/subject/TIP43/ses_1/TIP43_20250624_TIP43_68647ad3/"
                    "Sessions/Session_20250624120028978/Registrations/"
                    "LocatorRegistrations/BaseMarker20250624124756928.xml"),
            2: Path("/home/skarvela/mnt/hummel-data/TI/TI_EEG/EEG/"
                    "Neuronavigation/subject/TIP43/ses_2/Session_20250630103905099/"
                    "Registrations/LocatorRegistrations/BaseMarker20250630105330573.xml"),
        },
    },
}

# want to process whole recording or blocks?
whole_recording = True # set to True if you intend to process the whole recording instead of blocks

# ROOT PATHS
fif_root = Path("/home/skarvela/mnt/hummel-data/TI/TI_EEG/EEG/Preprocessed")
whole_root = Path("/home/skarvela/mnt/hummel-data/TI/TI_EEG/EEG/Preprocessed") # change according to position of whole recording
# /home/skarvela/mnt/hummel-data/TI/TI_EEG/EEG/Preprocessed/TIP39/ses_1/TIP39_ses1_task_final_preprocessed_raw_250.fif
nav_root = Path("/home/skarvela/mnt/hummel-data/TI/TI_EEG/EEG/Neuronavigation")
out_montage = Path("/mnt/data/Semester_Projects/michelet/Source_Localization_Davide/OUT_montages_run_pipeline1")
subjects_dir = Path("/mnt/data/Semester_Projects/michelet/Source_Localization_Davide/Freesurfer/Subjects")
coord_frame = "ras"

# Some HELPERS for the pipeline of montage extraction and personalization
def ask_list(prompt):
    raw = input(prompt).strip()
    return [int(x) for x in raw.split(",")]


def detect_available_blocks(subject, session):
    folder = (
        fif_root / subject / f"ses_{session}" /
        f"{subject}_ses{session}_task_blocks"
    )

    if not folder.exists():
        return []

    pattern = re.compile(rf"{subject}_ses{session}_task_block(\d+)\.fif")
    blocks = []

    for f in folder.iterdir():
        m = pattern.match(f.name)
        if m:
            blocks.append(int(m.group(1)))

    return sorted(blocks)


def find_nav_csv(subject, session):
    """
    Search for neuronavigation CSV in multiple possible folder structures.
    """
    candidate_dirs = [
        nav_root / "mni" / subject / f"ses_{session}",
        nav_root / "subject" / subject / f"ses_{session}",
        nav_root / subject / f"ses_{session}",
    ]

    candidate_names = [
        "Electrode_position_ras.csv",
        "Electrode_position_RAS.csv",
        "electrodes_ras.csv",
        "EEG_RAS.csv",
        "electrode_positions_ras.csv",
    ]

    for d in candidate_dirs:
        if not d.exists():
            continue
        for name in candidate_names:
            f = d / name
            if f.exists():
                logger.info(f"Found neuronavigation CSV: {f}")
                return f

    raise FileNotFoundError(
        f"Neuronavigation CSV not found for {subject} session {session}"
    )


def save_montage_visualization(montage, subject, session, out_root):
    out_dir = out_root / subject / "montages_vis"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_path = out_dir / f"{subject}_ses{session}_montage.png"

    try:
        fig = montage.plot(show=False)
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)
        logger.info(f"Saved montage visualization: {fig_path}")
    except Exception as e:
        logger.error(f"Failed to save montage visualization: {e}")

    return fig_path

if whole_recording:

    print("\n[MODE: WHOLE RECORDING]")
    print("Skipping block selection: in full recording mode there are no blocks to choose.\n")
    print("Available subjects:")
    for s in subject_config.keys():
        print("  -", s)

    user_subjects = (
        input("\nEnter subjects (comma separated): ").strip().replace(" ", "").split(",")
    )
    user_subjects = [s for s in user_subjects if s in subject_config]

    if not user_subjects:
        raise ValueError("No valid subjects selected.")

    # Surfaces once per subject
    for subject in user_subjects:
        make_scalp_surfaces(subject, subjects_dir)
        make_bem_watershed(subject, subjects_dir)

        print(f"\n-------- SUBJECT {subject} --------")
        sessions_available = subject_config[subject]["sessions"]
        print("Available sessions:", sessions_available)

        ses_list = ask_list("Select sessions: ")
        ses_list = [s for s in ses_list if s in sessions_available]

        if not ses_list:
            print("→ No valid sessions chosen, skipping subject.")
            continue

        for session in ses_list:
            print(f"\n---- SESSION {session} ----")
            fid_xml = subject_config[subject]["basemarker"][2 if session == 3 else session]
            # Neuronavigation CSV
            nav_csv = find_nav_csv(subject, 2 if session == 3 else session)
            # Path del FIF completo da processare
            fif_whole = (
                whole_root / subject / f"ses_{session}" /
                f"{subject}_ses{session}_task_final_preprocessed_raw_250.fif"
            )

            print(f"Using full FIF:\n  {fif_whole}")
            if not fif_whole.exists():
                logger.error(f"Full FIF not found: {fif_whole}")
                continue

            # load and apply montage to whole recording of the session
            logger.info(f"Computing montage for {subject} session {session}")
            raw = mne.io.read_raw_fif(fif_whole, preload=True)

            raw_montaged = personalize_montage(
                fif_path=fif_whole,
                neuronavigation_path=nav_csv,
                fiducials_path=fid_xml,
                output_dir=out_montage,
                subject=subject,
                session=session,
                block=0,
                coord_frame=coord_frame,
                plot=False
            )

            montage = raw_montaged.get_montage()

            # save the png image of the montage
            save_montage_visualization(montage, subject, session, out_root=out_montage)
            # out_fif = (
            #     out_montage / subject / f"ses_{session}" /
            #     f"{subject}_ses{session}_whole_personalized_montage.fif"
            # )
            # out_fif.parent.mkdir(parents=True, exist_ok=True)
            # raw.set_montage(montage, on_missing="ignore")
            # raw.save(out_fif, overwrite=True)
            #logger.info(f"✅ Saved FULL recording FIF → {out_fif}")

    print("\n✅ Full-mode pipeline finished.\n")

else:
    # BLOCK BASED MODE
    print("\n[MODE: BLOCK RECORDINGS]")
    print("Available subjects:")
    for s in subject_config.keys():
        print("  -", s)

    user_subjects = (
        input("\nEnter subjects (e.g. TIP39,TIP40): ").strip().replace(" ", "").split(",")
    )

    user_subjects = [s for s in user_subjects if s in subject_config]

    if not user_subjects:
        raise ValueError("No valid subjects selected.")

    for subject in user_subjects:

        print(f"\n-------- SUBJECT {subject} --------")
        sessions_available = subject_config[subject]["sessions"]
        print("Available sessions:", sessions_available)

        ses_list = ask_list("Select sessions: ")
        ses_list = [s for s in ses_list if s in sessions_available]

        if not ses_list:
            print("Skipping subject.")
            continue

        for session in ses_list:

            print(f"\n---- SESSION {session} ----")
            # detect blocks
            available_blocks = detect_available_blocks(subject, session)
            print("Blocks available:", available_blocks)

            # STILL INTERACTIVE ONLY IN BLOCK MODE
            blocks = ask_list("Select blocks: ")
            blocks = [b for b in blocks if b in available_blocks]

            if not blocks:
                print("No valid blocks. Skipping session.")
                continue

            fid_xml = subject_config[subject]["basemarker"][2 if session == 3 else session]
            nav_csv = find_nav_csv(subject, 2 if session == 3 else session)

            first_fif_to_localize = (
                fif_root / subject / f"ses_{session}" / f"{subject}_ses{session}_task_blocks" /
                f"{subject}_ses{session}_task_block{blocks[0]}.fif"
            )

            logger.info(f"Computing montage for {subject} session {session}")

            raw_subset = personalize_montage(
                fif_path=first_fif_to_localize,
                neuronavigation_path=nav_csv,
                fiducials_path=fid_xml,
                output_dir=out_montage,
                subject=subject,
                session=session,
                block=blocks[0],
                coord_frame=coord_frame,
                plot=False
            )

            montage = raw_subset.get_montage()

            save_montage_visualization(
                montage=montage,
                subject=subject,
                session=session,
                out_root=out_montage
            )

            make_scalp_surfaces(subject, subjects_dir)
            make_bem_watershed(subject, subjects_dir)

            for block in blocks:
                fif_path = (
                    fif_root / subject / f"ses_{session}" /
                    f"{subject}_ses{session}_task_blocks" /
                    f"{subject}_ses{session}_task_block{block}.fif"
                )

                logger.info(f"Processing block {block} for {subject} ses {session}")

                try:
                    raw = mne.io.read_raw_fif(fif_path, preload=True)
                    raw.set_montage(montage, on_missing="ignore")
                    out_fif = (
                        out_montage / subject / f"ses_{session}" /
                        f"{subject}_ses{session}_block{block}_personalized_montage.fif"
                    )
                    out_fif.parent.mkdir(parents=True, exist_ok=True)
                    raw.save(out_fif, overwrite=True)

                    logger.info(f"✅ Saved block FIF with montage → {out_fif}")

                except Exception as e:
                    logger.error(f"Failed block {block}: {e}")
                    continue

    print("\n✅ BLOCK-RECORDING pipeline finished ✅\n")