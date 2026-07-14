# Author Stavriani Skarvelaki / Maelys Clerget

"""
utils06.py

Utilities for EEG source localization STEP 2 (after coregistration):

1. Creation of the forward solution (both surface level and volume based)
2. Inverse solution computation
3. Extraction of the source time courses according to a specified atlas

Notes:
- Assumes that run_pipeline1.py has already been executed, and thus already have the personalized montage and the BEM surfaces..
- These functions are thought to be called inside a higher-level pipeline script
  that loops over subjects / sessions / blocks.
"""

import mne
import numpy as np
import csv
from pathlib import Path
import logging
from copy import deepcopy
import os.path as op
import nibabel as nib

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# STRIATUM LABELS FOR VOLUME SOURCE SPACE (CHANGE IF NEEDED based on aseg labels)
STRIATUM_LABELS = [
    "Left-Caudate", "Right-Caudate",
    "Left-Putamen", "Right-Putamen",
    "Left-Accumbens-area", "Right-Accumbens-area",
]
# STRIATUM_LABELS_EXTENDED = STRIATUM_LABELS + [
#     "Left-VentralDC", "Right-VentralDC",
# ] #can include VentralDC if needed as well for striatum

# SOURCE SPACES: surface, volume AND mixed (3 functions)
def create_surface_source_space(subject, subjects_dir, spacing="oct6"):
    logger.info(f"Creating surface source space ({spacing})")
    print(f"🕒 Creating surface source space ({spacing})")
    return mne.setup_source_space(
        subject,
        spacing=spacing,
        add_dist=False,
        subjects_dir=subjects_dir,
    )

def create_striatum_volume_source_space(subject, subjects_dir, bem=None, pos=5.0):
    """
    Volume source space restricted to the STRIATUM (Caudate, Putamen, Accumbens).
    """
    aseg = f"{subjects_dir}/{subject}/mri/aseg.mgz"

    all_labels = mne.get_volume_labels_from_aseg(aseg)
    # uncomment to look at the aseg labels and select the labels of the desired deep brain regions (based on the labels, we can select the STRIATUM_LABELS to consider)
    # print("Available volume labels in aseg:", all_labels)
    # raise SystemExit("Debug stop: inspect the labels above.")
    labels_vol = [lab for lab in all_labels if lab in STRIATUM_LABELS]
    logger.info(f"Using striatal structures: {labels_vol}")
    print(f"🕒 Creating volume source space for striatum structures: {labels_vol}")
    src = mne.setup_volume_source_space(
        subject=subject,
        mri=aseg,
        pos=pos,
        bem=bem,
        volume_label=labels_vol,
        subjects_dir=subjects_dir,
        add_interpolator=False
    )
    return src

def create_mixed_source_space(subject, subjects_dir, bem=None, spacing="oct6", pos=5.0):
    print("🕒 Creating mixed source space (cortex + striatum)")
    surf = create_surface_source_space(subject, subjects_dir, spacing)
    vol  = create_striatum_volume_source_space(subject, subjects_dir, bem, pos)
    logger.info("Mixed source space: cortex + striatum")
    return surf + vol

# Create BEM model
def create_bem_solution(subject, subjects_dir, conductivity=(0.3, 0.006, 0.3), ico=4):
    model = mne.make_bem_model(
        subject=subject,
        ico=ico,
        conductivity=conductivity,
        subjects_dir=subjects_dir
    )
    bem = mne.make_bem_solution(model)
    logger.info("BEM solution created.")
    return bem

# FORWARD SOLUTION
def create_forward_solution(info, trans, src, bem, mindist=5.0):
    logger.info("🕒 Computing forward solution...")
    fwd = mne.make_forward_solution(
        info=info,
        trans=trans,
        src=src,
        bem=bem,
        eeg=True,
        meg=False,
        mindist=mindist,
        n_jobs=1
    )
    logger.info("✅ Forward solution created.")
    return fwd

# ORIENTATIONS of diapoles in the forward solution (normal to surface for close ones, free for the volume)
def orient_forward_solution(fwd, mode):
    """
    Takes in the forward solution and for the different modes:
    "Surface" → constrain dipoles normal to surface
    "Volume" or "Mixed"  → free orientations
    """
    if mode == "surface":
        return mne.convert_forward_solution(
            fwd, surf_ori=True, force_fixed=True, use_cps=True
        )
    else:
        return fwd # orientations in the mixed will be handled later in the make inverse operator step
    
    
def _surface_hemi_name(src_entry, src_index: int) -> str:
    """Return lh/rh for a surface source-space entry when possible."""
    if src_index == 0:
        return "lh"
    if src_index == 1:
        return "rh"
    return ""


def _surface_vertex_to_label(subject_fs: str, subjects_dir: Path, parc: str = "aparc") -> dict[tuple[str, int], str]:
    """Map FreeSurfer surface vertices to aparc label names."""
    vertex_to_label = {}
    # reads anatomical labels from freesurfer 
    for hemi in ("lh", "rh"):
        for label in mne.read_labels_from_annot(
            subject_fs,
            parc=parc,
            hemi=hemi,
            subjects_dir=subjects_dir,
        ):
            # loops over every vertex belonging to that brain region.
            for vertex in label.vertices:
                vertex_to_label[(hemi, int(vertex))] = label.name # ex: vertex_to_label[("lh", 12345)] = "precentral-lh"
    return vertex_to_label


def save_dipole_area_index(
    fwd,
    mode: str,
    subject_fs: str,
    subjects_dir: Path,
    out_dir: Path,
    base_tag: str,
) -> Path:
    """Save a CSV mapping each plotted leadfield column to its anatomical area."""
    fwd_oriented = orient_forward_solution(fwd, mode)
    leadfield = fwd_oriented["sol"]["data"] # get leadfield matrix
    src = fwd_oriented["src"] # get the source space
    total_sources = sum(src_entry["nuse"] for src_entry in src) #how many anatomical source points exist 
    n_orient = leadfield.shape[1] // total_sources #computes how many orientations each source has. fixed orientation = 1 OR free orientation = 3
    if leadfield.shape[1] % total_sources != 0: #checks that the leadfield columns match the source count
        raise RuntimeError(
            f"Leadfield columns ({leadfield.shape[1]}) are not compatible with "
            f"source-space points ({total_sources})."
        )

    vertex_to_label = {}
    if mode in {"surface", "mixed"}:
        vertex_to_label = _surface_vertex_to_label(subject_fs, subjects_dir) # reads FreeSurfer aparc labels

    csv_path = out_dir / f"{base_tag}_dipole_index_to_area.csv"
    orientation_names = ("x", "y", "z") if n_orient == 3 else tuple(str(i) for i in range(n_orient)) 

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "dipole_index",
                "source_index",
                "orientation",
                "source_space",
                "hemisphere",
                "vertex",
                "voxel",
                "area",
            ],
        )
        writer.writeheader()

        source_index = 0
        #loops over each source-space block
        for src_index, src_entry in enumerate(src): 
            is_volume = src_entry.get("seg_name") is not None #check wether this source space block is volume. If it is volume, the anatomical area comes from seg_name.
            source_space = "volume" if is_volume else "surface"
            hemisphere = "" if is_volume else _surface_hemi_name(src_entry, src_index) #empty if is volume else lh or rh 
            area = src_entry.get("seg_name", "") if is_volume else "" #if is volume --> seg name else empty 

            # Loop through every source point inside that source-space block.
            for vertex in src_entry["vertno"]:
                vertex = int(vertex)
                if not is_volume:
                    area = vertex_to_label.get((hemisphere, vertex), "unknown") #if is surface, get the area name from the vertex_to_label mapping. ex: ("lh", 101) -> precentral-lh

                for orientation_index in range(n_orient): # for this same brain location, write one row per orientation
                    dipole_index = source_index * n_orient + orientation_index
                    writer.writerow(
                        {
                            "dipole_index": dipole_index,
                            "source_index": source_index,
                            "orientation": orientation_names[orientation_index],
                            "source_space": source_space,
                            "hemisphere": hemisphere,
                            "vertex": "" if is_volume else vertex,
                            "voxel": vertex if is_volume else "",
                            "area": area,
                        }
                    )

                source_index += 1

    print(f"Saved dipole-to-area index CSV: {csv_path}")
    return csv_path


# INVERSE OPERATOR
def create_inverse_operator(info, fwd, noise_cov, mode):
    if mode == "surface":
        inv = mne.minimum_norm.make_inverse_operator(
            info,
            fwd,
            noise_cov,
            loose=0.2, # consider the oriented version of the surface forward solution
            depth=0.8
        )
    else:  # mode == "volume" or "mixed" 
        inv = mne.minimum_norm.make_inverse_operator(
            info,
            fwd,
            noise_cov,
            loose=1.0, # free orientation in the volume, but also in the mixed because can't handle separately the spaces
            depth=0.8 # depth compensation always applied
        )
    logger.info("Inverse operator created.")
    return inv

# APPLY INVERSE operator to find the source time courses (STC)
def compute_stc(raw, inverse_operator, lambda2, method):
    stc = mne.minimum_norm.apply_inverse_raw(
        raw,
        inverse_operator,
        lambda2=lambda2, 
        method=method
    )
    logger.info("STC computed.")
    return stc


# MORPHING
def compute_surface_source_morph(stc, subject_from, subjects_dir):
    """
    Compute a reusable cortical surface morph from one subject to fsaverage.

    Returns
    -------
    morph : mne.SourceMorph
        The reusable morph object.
    src_to : mne.SourceSpaces
        The target source space used by the morph.
    """
    subjects_dir = Path(subjects_dir)
    fsaverage_dir = subjects_dir / "fsaverage"
    if not fsaverage_dir.exists():
        raise FileNotFoundError(
            f"Target FreeSurfer subject not found: {fsaverage_dir}. "
            "Install or copy fsaverage into SUBJECTS_DIR before morphing."
        )

    logger.info("Creating fsaverage ico5 source space for morphing.")
    # Create the target fsaverage surface source space where the morphed STC will be defined
    src_to = mne.setup_source_space(
        "fsaverage",
        spacing="ico5",
        add_dist=False,
        subjects_dir=subjects_dir,
    )

    logger.info("Computing surface source morph: %s -> fsaverage", subject_from)
    # Compute the mapping from the subject's source space to the fsaverage target source space.
    morph = mne.compute_source_morph(
        stc,
        subject_from=subject_from,
        subject_to="fsaverage",
        src_to=src_to,
        subjects_dir=subjects_dir,
    )
    return morph, src_to


def apply_source_morph(stc, morph):
    """Apply a precomputed source morph to one source estimate."""
    logger.info("Applying source morph to STC.")
    return morph.apply(stc)



# SURFACE CASE
def extract_surface_label_ts(stc, src_all, subject, subjects_dir, parc="aparc"):
    if isinstance(src_all, list):
        src_all = mne.SourceSpaces(src_all) 
    
    labels_lh = mne.read_labels_from_annot(subject, parc=parc, hemi="lh", subjects_dir=subjects_dir)
    labels_rh = mne.read_labels_from_annot(subject, parc=parc, hemi="rh", subjects_dir=subjects_dir)
    labels = labels_lh + labels_rh

    ts = mne.extract_label_time_course(
        stc,
        labels,
        src=src_all,
        mode="mean_flip",
        allow_empty=True,
    )
    return labels, ts

# VOLUME CASE
def extract_volume_roi_ts(stc, src_all, roi_names):
    """
    Extract time series for subcortical volumetric ROIs defined 
    by src_vol (each entry corresponds to a structure).
    """
    if isinstance(src_all, list): 
        src_all = mne.SourceSpaces(src_all)

    data = stc.data
    n_rows, n_times = data.shape
    total_nuse = sum(s["nuse"] for s in src_all)
    if n_rows % total_nuse != 0:
        raise RuntimeError(
            f"STC rows ({n_rows}) isn't multiple of total_nuse ({total_nuse}). "
            "There's a mismatch STC ↔ src (probably not using fwd['src'])."
        )

    ori_factor = n_rows // total_nuse  # 1 (fixed) o 3 (free orientation)

    roi_ts = []
    names_out = []

    row_start = 0
    for s in src_all:
        nuse = s["nuse"]
        seg_name = s.get("seg_name", None)
        n_rows_this = nuse * ori_factor
        row_end = row_start + n_rows_this

        block = data[row_start:row_end, :]  # (n_rows_this, n_times)

        if seg_name in roi_names:
            # Reshape to separate orientations
            if ori_factor > 1:
                block = block.reshape(nuse, ori_factor, n_times)
                # norm over x,y,z components
                block = np.linalg.norm(block, axis=1)
            else:
                block = block.reshape(nuse, n_times)

            # Mean over dipoles of that structure → (n_times,)
            roi_ts.append(block.mean(axis=0))
            names_out.append(seg_name)

        row_start = row_end

    if len(roi_ts) == 0:
        ts = np.empty((0, n_times))
    else:
        ts = np.vstack(roi_ts)  # (n_roi, n_times)

    return names_out, ts

# HANDLE ALL CASES and extract the corresponding labels based on the modality choice
def extract_labels_all_modes(stc, src_all, mode, subject, subjects_dir):
    """
    Unified extraction of time series for:
    - mode == "surface" → cortex only (aparc)
    - mode == "volume"  → striatum only (STRIATUM_LABELS)
    - mode == "mixed"   → cortex + striatum
    """
    # Always use the forward source space:
    # src_all = fwd['src'] in the main script, to avoid having index issues due to MNE internal handling.
    if isinstance(src_all, list):
        src_all = mne.SourceSpaces(src_all)

    # SURFACE ONLY
    if mode == "surface":
        labels, ts = extract_surface_label_ts(
            stc, src_all, subject, subjects_dir, parc="aparc"
        )
        return labels, ts, "surface"

    # VOLUME ONLY (striatum)
    elif mode == "volume":
        roi_names, ts = extract_volume_roi_ts(
            stc, src_all, STRIATUM_LABELS
        )
        return roi_names, ts, "volume"

    # MIXED: surface + volume (striatum)
    elif mode == "mixed":
        # first cortex
        labels_surf, ts_surf = extract_surface_label_ts(
            stc, src_all, subject, subjects_dir, parc="aparc"
        )

        # and then Striatum
        roi_names_vol, ts_vol = extract_volume_roi_ts(
            stc, src_all, STRIATUM_LABELS
        )

        return {
            "surface_labels": labels_surf,
            "surface_ts": ts_surf,
            "volume_labels": roi_names_vol,
            "volume_ts": ts_vol,
        }, None, "mixed"

    else:
        raise ValueError("mode must be 'surface', 'volume', or 'mixed'")

def save_label_ts_as_fif(roi_ts, roi_names, stc, fname):
    """
    Saves ROI time series (n_roi × n_times) as Evoked .fif.
    If there is a mismatch between the number of rows in roi_ts and the number of names in roi_names,
    it safely truncates to ensure consistency.
    """
    n_roi_ts, n_times = roi_ts.shape
    n_names = len(roi_names)

    if n_roi_ts != n_names:
        print(f"⚠️ Mismatch between n_roi_ts={n_roi_ts} and n_names={n_names} for {fname.name}. "
              "Automatically adapting (cropping).")
        if n_roi_ts > n_names:
            # More rows than names, then keep only the first n_names
            roi_ts = roi_ts[:n_names, :]
        else:
            # More names than rows, then keep only the first n_roi_ts
            roi_names = roi_names[:n_roi_ts]

        n_roi_ts, _ = roi_ts.shape
        n_names = len(roi_names)
        print(f"After adaptation: n_roi_ts={n_roi_ts}, n_names={n_names}")

    info = mne.create_info(
        ch_names=roi_names,
        sfreq=1.0 / stc.tstep,
        ch_types="misc",
    )

    evoked = mne.EvokedArray(
        data=roi_ts,
        info=info,
        tmin=stc.tmin,
        comment="ROI time series"
    )

    evoked.save(fname, overwrite=True)
    return evoked