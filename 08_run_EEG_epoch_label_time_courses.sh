#!/bin/bash
# Author Stavriani Skarvelaki / Maelys Clerget

#SBATCH --job-name=eeg_epoch_labels
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=10
#SBATCH --error=/home/clerget/Scripts/preprocessing_pipeline/logs/%x_%A_%a.err
#SBATCH --output=/home/clerget/Scripts/preprocessing_pipeline/logs/%x_%A_%a.out

# Requested memory should stay <= requested CPUs x 7000 MB.

umask 0002

module purge
module load gcc python

cd /work/uphummel/studies/tTIS-EEG/code/Maelys/

PARAM_LIST="/work/uphummel/studies/tTIS-EEG/code/Maelys/files_for_epoch_label_time_courses.txt"
CURRENT_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAM_LIST")

if [ -z "$CURRENT_FILE" ]; then
    echo "Error: No file found for Array ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

read -r SUB SES TASK MODE METHOD COV_LABEL LABEL_KIND CONNECTIVITY_BASELINE <<< "$CURRENT_FILE"

if [ -z "$MODE" ]; then
    MODE="surface"
fi

if [ -z "$METHOD" ]; then
    METHOD="MNE"
fi

if [ -z "$COV_LABEL" ]; then
    COV_LABEL="15"
fi

if [ -z "$LABEL_KIND" ]; then
    LABEL_KIND="labels"
fi

if [ -z "$CONNECTIVITY_BASELINE" ]; then
    CONNECTIVITY_BASELINE="stim"
fi

DERIV_ROOT="/work/uphummel/studies/tTIS-EEG/derivatives/EEG"
BASE_TAG="${SUB}_ses${SES}_task-${TASK}_src-${MODE}_method-${METHOD}"

if [ "$TASK" = "task" ]; then
    COV_TAG="$COV_LABEL"
    if [[ "$COV_TAG" != s* ]]; then
        COV_TAG="s${COV_TAG}"
    fi

    LABELS_DIR="$DERIV_ROOT/sub-$SUB/ses-$SES/source_reconstruction/inverse_solution/$MODE/$METHOD/cov-$COV_TAG"
    LABELS_FIF="$LABELS_DIR/${BASE_TAG}_cov-${COV_TAG}_${LABEL_KIND}.fif"
    if [ ! -f "$LABELS_FIF" ]; then
        ALT_LABELS_DIR="$DERIV_ROOT/sub-$SUB/ses-$SES/source_reconstruction/inverse_solution/$MODE/$METHOD/task/cov$COV_TAG"
        ALT_LABELS_FIF="$ALT_LABELS_DIR/${BASE_TAG}_cov-${COV_TAG}_${LABEL_KIND}.fif"
        if [ -f "$ALT_LABELS_FIF" ]; then
            LABELS_DIR="$ALT_LABELS_DIR"
            LABELS_FIF="$ALT_LABELS_FIF"
        fi
    fi
else
    LABELS_DIR="$DERIV_ROOT/sub-$SUB/ses-$SES/source_reconstruction/inverse_solution/$MODE/$METHOD/$TASK"
    LABELS_FIF="$LABELS_DIR/${BASE_TAG}_${LABEL_KIND}.fif"
fi

echo "SUB = '$SUB'"
echo "SES = '$SES'"
echo "TASK = '$TASK'"
echo "MODE = '$MODE'"
echo "METHOD = '$METHOD'"
echo "COV_LABEL = '$COV_LABEL'"
echo "LABEL_KIND = '$LABEL_KIND'"
echo "CONNECTIVITY_BASELINE = '$CONNECTIVITY_BASELINE'"
echo "LABELS_FIF = '$LABELS_FIF'"
echo "Processing ROI label epochs for sub-$SUB ses-$SES task-$TASK mode-$MODE method-$METHOD label-$LABEL_KIND"

if [ ! -f "$LABELS_FIF" ]; then
    echo "Error: labels FIF not found: $LABELS_FIF"
    exit 1
fi

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

        python 08_EEG_epoch_label_time_courses.py \
            --labels-fif "'"$LABELS_FIF"'" \
            --subject "'"$SUB"'" \
            --session "'"$SES"'" \
            --task "'"$TASK"'" \
            --mode "'"$MODE"'" \
            --method "'"$METHOD"'" \
            --cov-label "'"$COV_LABEL"'" \
            --connectivity-baseline "'"$CONNECTIVITY_BASELINE"'"
    '
