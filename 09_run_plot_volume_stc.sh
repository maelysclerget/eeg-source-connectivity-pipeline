#!/bin/bash
# Author Stavriani Skarvelaki / Maelys Clerget

#SBATCH --job-name=plot_volume_stc
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=29
#SBATCH --error=/home/clerget/Scripts/preprocessing_pipeline/logs/%x_%j.err
#SBATCH --output=/home/clerget/Scripts/preprocessing_pipeline/logs/%x_%j.out

# Requested memory should stay <= requested CPUs x 7000 MB.

umask 0002

module purge
module load gcc python

cd /home/clerget/Scripts/code

source /work/uphummel/shared/software/python_envs/mne_venv/bin/activate

python 09_plot_volume_stc.py
