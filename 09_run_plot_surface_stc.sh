#!/bin/bash
# Author Stavriani Skarvelaki / Maelys Clerget

#SBATCH --job-name=plot_volume_stc
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10
#SBATCH --error=/home/clerget/Scripts/preprocessing_pipeline/logs/%x_%j.err
#SBATCH --output=/home/clerget/Scripts/preprocessing_pipeline/logs/%x_%j.out

umask 0002

module purge
module load gcc python

cd /home/clerget/Scripts/code

source /work/uphummel/shared/software/python_envs/mne_venv/bin/activate

python 09_plot_surface_stc.py

