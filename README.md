# EEG Source Connectivity Pipeline

Python and SLURM scripts for EEG source reconstruction and ROI-level source-space connectivity analysis.

The pipeline prepares individualized EEG montages, builds BEM and forward models, computes inverse source estimates, extracts ROI time courses, epochs those time courses, and estimates ROI-to-ROI connectivity across canonical frequency bands.

## Repository Layout

```text
.
├── 05_EEG_montage_BEM.py                 # Personalized montage, scalp surfaces, BEM preparation
├── 06_EEG_forward_solution.py            # Surface/volume/mixed forward solutions
├── 06_EEG_inverse_solution_only.py       # Inverse solutions and ROI time courses
├── 07_EEG_connectivity.py                # ROI-to-ROI connectivity matrices and plots
├── 08_EEG_epoch_label_time_courses.py    # Epoch ROI time courses before connectivity
├── ICA/                                  # ICA and preprocessing utilities
├── config/                               # Input lists for batch/HPC runs
├── scripts/slurm/                        # SLURM run, submit, and debug scripts
├── plots/                                # Plotting and visualization scripts
├── examples/figures/                     # Lightweight example outputs for GitHub
├── docs/                                 # Extra documentation
├── archive/                              # Older exploratory versions kept for reference
└── outputs/                              # Local/generated outputs, ignored by Git
```

## Pipeline Steps

1. **Preprocess EEG / ICA**
   - Main scripts live in `ICA/`.
   - Outputs should be saved under the derivatives directory used by the study.

2. **Create personalized montage and BEM**
   - Python: `05_EEG_montage_BEM.py`
   - SLURM: `scripts/slurm/05_run_EEG_montage_BEM.sh`
   - Uses neuronavigation coordinates, FreeSurfer outputs, and preprocessed FIF files.

3. **Compute forward solution**
   - Python: `06_EEG_forward_solution.py`
   - SLURM: `scripts/slurm/06_run_EEG_forward_solution.sh`
   - Supports `surface`, `volume`, and `mixed` source spaces.

4. **Compute inverse solution and ROI time courses**
   - Python: `06_EEG_inverse_solution_only.py`
   - SLURM: `scripts/slurm/06_run_EEG_inverse_solution_only.sh`
   - Supports inverse methods such as MNE, sLORETA, and eLORETA.

5. **Epoch ROI label time courses**
   - Python: `08_EEG_epoch_label_time_courses.py`
   - SLURM: `scripts/slurm/08_run_EEG_epoch_label_time_courses.sh`
   - Creates task and resting-state epoch files from continuous ROI time courses.

6. **Compute source connectivity**
   - Python: `07_EEG_connectivity.py`
   - SLURM: `scripts/slurm/07_run_EEG_connectivity.sh`
   - Saves ROI connectivity matrices, heatmaps, circle plots, and summary CSVs.

## Requirements

This project depends on the scientific Python/MNE ecosystem:

- Python 3.10+
- `mne`
- `mne-connectivity`
- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `seaborn`
- FreeSurfer, for BEM/source-space workflows
- Apptainer/Singularity and SLURM, for the provided cluster scripts

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Running a Single Subject

Most scripts accept command-line arguments for subject/session/task selection. For example:

```bash
python 06_EEG_forward_solution.py \
  --subject 41Y01 \
  --session 1 \
  --mode surface
```

For HPC runs, update the paths in `scripts/slurm/*.sh` and the file lists in `config/`, then submit the corresponding job script.

## Data Notes

Large scientific data files should not be committed to GitHub. The `.gitignore` excludes common EEG/MRI formats and generated output folders, including:

- raw and derivative data (`data/`, `derivatives/`, `results/`, `outputs/`)
- MNE/FIF and MRI files (`*.fif`, `*.stc`, `*.nii`, `*.mgz`, etc.)
- generated logs, caches, and Python bytecode

Keep the repository focused on code, configuration examples, and documentation.

## Path Configuration

Several scripts currently use absolute project paths for the cluster environment, such as:

```text
/work/uphummel/studies/tTIS-EEG/
```

Before running the pipeline in another environment, update these constants and the matching paths inside `scripts/slurm/`.

## Suggested GitHub Workflow

1. Keep source code and lightweight config files tracked.
2. Keep generated figures, FIF files, MRI derivatives, and logs ignored.
3. Add example input lists in `config/` when they are safe to share.
4. Document any machine-specific path changes in `docs/`.
