# Project Structure

This repository is organized so the main analysis stages remain easy to find while support files are grouped by purpose.

## Main Pipeline Scripts

- `05_EEG_montage_BEM.py`: creates personalized EEG montage files and BEM assets.
- `06_EEG_forward_solution.py`: builds forward solutions for source reconstruction.
- `06_EEG_inverse_solution_only.py`: computes inverse solutions and ROI time courses.
- `08_EEG_epoch_label_time_courses.py`: epochs ROI label time courses.
- `07_EEG_connectivity.py`: computes source-space ROI connectivity.

## Supporting Folders

- `ICA/`: preprocessing and ICA-related code.
- `scripts/slurm/`: SLURM run, submit, and debug scripts.
- `config/`: small text files listing subjects/sessions/tasks for batch jobs.
- `plots/`: scripts for visual inspection and plotting source estimates/ROI time courses.
- `examples/figures/`: selected generated figures kept as lightweight examples for GitHub.
- `archive/`: older or exploratory versions kept for reference.
- `outputs/`: generated local output; this folder is ignored by Git.

## Notes For Future Cleanup

The next improvement would be to move reusable helpers into a Python package such as `src/eeg_source_connectivity/`. That should be done together with import updates and a quick execution test on the cluster, because the current scripts rely on direct local imports like `from utils06 import ...`.
