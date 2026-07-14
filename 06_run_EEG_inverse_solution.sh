#!/bin/bash
# Author Stavriani Skarvelaki / Maelys Clerget

# Resources chosen to optimize memory usage while respecting the cluster rule:
#SBATCH --job-name=eeg_inverse
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --mem=192G
#SBATCH --cpus-per-task=29
#SBATCH --error=/home/clerget/Scripts/preprocessing_pipeline/logs/%x_%A_%a.err
#SBATCH --output=/home/clerget/Scripts/preprocessing_pipeline/logs/%x_%A_%a.out

# Requested memory should stay <= requested CPUs x 7000 MB. 

umask 0002

module purge
module load gcc python

cd /work/uphummel/studies/tTIS-EEG/code/Maelys/

PARAM_LIST="/work/uphummel/studies/tTIS-EEG/code/Maelys/files_for_inverse_solution.txt"
CURRENT_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAM_LIST")

if [ -z "$CURRENT_FILE" ]; then
    echo "Error: No file found for Array ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

read -r SUB SES TASK MODE METHOD COV_DURATION MORPH_TO <<< "$CURRENT_FILE"

if [ -z "$MODE" ]; then
    MODE="surface"
fi

if [ -z "$METHOD" ]; then
    METHOD="sLORETA"
fi

if [ -z "$COV_DURATION" ]; then
    COV_DURATION="20.0"
fi

MORPH_ARGS=""
if [ "$MORPH_TO" = "fsaverage" ]; then
    MORPH_ARGS="--morph-to-fsaverage"
fi

echo "SUB = '$SUB'"
echo "SES = '$SES'"
echo "TASK = '$TASK'"
echo "MODE = '$MODE'"
echo "METHOD = '$METHOD'"
echo "COV_DURATION = '$COV_DURATION'"
echo "MORPH_TO = '$MORPH_TO'"
echo "Processing inverse solution for sub-$SUB ses-$SES task-$TASK mode-$MODE method-$METHOD cov-${COV_DURATION}s"

MNE_FS_IMG="/work/uphummel/shared/software/containers/mne_freesurfer/mne_freesurfer_8.1.sif"
FS_LICENSE="/work/uphummel/shared/software/containers/license.txt"
SUBJECTS_DIR="/work/uphummel/studies/tTIS-EEG/derivatives/MRI/freesurfer"

apptainer exec \
    --bind /work/uphummel:/work/uphummel \
    --bind /home/skarvela:/home/skarvela \
    --env FS_LICENSE="$FS_LICENSE" \
    --env SUBJECTS_DIR="$SUBJECTS_DIR" \
    "$MNE_FS_IMG" \
    bash -lc '
        export FREESURFER_HOME=/usr/local/freesurfer/8.1.0-1
        export PYTHONPATH=/work/uphummel/studies/tTIS-EEG/code/shared_python_packages_py313:$PYTHONPATH

        cd /home/clerget/Scripts/code

        echo "FREESURFER_HOME=$FREESURFER_HOME"
        echo "SUBJECTS_DIR=$SUBJECTS_DIR"

        python 06_EEG_inverse_solution.py \
            --subject "'"$SUB"'" \
            --session "'"$SES"'" \
            --task "'"$TASK"'" \
            --mode "'"$MODE"'" \
            --method "'"$METHOD"'" \
            --cov-duration "'"$COV_DURATION"'" \
            '"$MORPH_ARGS"'
    '
