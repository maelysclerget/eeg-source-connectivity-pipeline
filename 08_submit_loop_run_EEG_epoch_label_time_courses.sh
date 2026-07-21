#!/bin/bash
# Author Stavriani Skarvelaki / Maelys Clerget

# Submit ROI label epoching jobs as a SLURM array.
# This scans raw EEG files, checks for step 06 label time-course FIF outputs,
# skips already-created step 08 Epochs FIF files, and submits one array row per
# missing subject/session/task/mode/method/covariance/label-kind combination.

DATA_ROOT="/work/uphummel/studies/tTIS-EEG/data/raw/EEG"
DERIV_ROOT="/work/uphummel/studies/tTIS-EEG/derivatives/EEG"

LOG_DIR="/work/uphummel/studies/tTIS-EEG/code/Maelys/logs"
PARAM_LIST="/work/uphummel/studies/tTIS-EEG/code/Maelys/files_for_epoch_label_time_courses.txt"
RUN_SCRIPT="/home/clerget/Scripts/code/08_run_EEG_epoch_label_time_courses.sh"

# Source space modes and inverse methods to epoch.
MODES=("surface" "volume" "mixed")
METHODS=("MNE" "sLORETA" "eLORETA")

# For task data, epoch both covariance outputs from step 06.
TASK_COV_LABELS=("4" "15")

# This controls which task baseline is copied to the compatibility
# *_baseline-epo.fif file used by connectivity when --baseline-kind baseline.
CONNECTIVITY_BASELINE="stim"

# Optional cap for complete nonbaseline epochs per task block. Use "none" to
# keep all complete epochs and let the Python script check that blocks align.
N_EPOCHS_PER_BLOCK="none"

mkdir -p "$LOG_DIR"
> "$PARAM_LIST"

echo "Scanning $DATA_ROOT for task-task and task-RSpre EEG files."

find "$DATA_ROOT" -type f -name "sub-*_ses-*_task-*_eeg.vhdr" | sort | while read -r FILE
do
    BASENAME=$(basename "$FILE")

    SUB=$(echo "$BASENAME" | sed -E 's/^sub-([^_]+)_ses-.*/\1/')
    SES=$(echo "$BASENAME" | sed -E 's/^sub-[^_]+_ses-([^_]+)_task-.*/\1/')
    TASK=$(echo "$BASENAME" | sed -E 's/^sub-[^_]+_ses-[^_]+_task-([^_]+)_eeg\.vhdr$/\1/')

    if [ "$TASK" != "task" ] && [ "$TASK" != "RSpre" ]; then
        continue
    fi

    for MODE in "${MODES[@]}"
    do
        for METHOD in "${METHODS[@]}"
        do
            COV_LABELS=("none")
            if [ "$TASK" = "task" ]; then
                COV_LABELS=("${TASK_COV_LABELS[@]}")
            fi

            for COV_LABEL in "${COV_LABELS[@]}"
            do
                if [ "$TASK" = "task" ]; then
                    COV_TAG="$COV_LABEL"
                    if [[ "$COV_TAG" != s* ]]; then
                        COV_TAG="s${COV_TAG}"
                    fi
                    INVERSE_DIR="$DERIV_ROOT/sub-$SUB/ses-$SES/source_reconstruction/inverse_solution/$MODE/$METHOD/cov-$COV_TAG"
                    BASE_TAG="${SUB}_ses${SES}_task-${TASK}_src-${MODE}_method-${METHOD}_cov-${COV_TAG}"
                    PARAM_COV_LABEL="$COV_LABEL"
                else
                    INVERSE_DIR="$DERIV_ROOT/sub-$SUB/ses-$SES/source_reconstruction/inverse_solution/$MODE/$METHOD/$TASK"
                    BASE_TAG="${SUB}_ses${SES}_task-${TASK}_src-${MODE}_method-${METHOD}"
                    PARAM_COV_LABEL="none"
                fi

                if [ "$MODE" = "mixed" ]; then
                    LABEL_KINDS=("surface_labels" "volume_labels")
                else
                    LABEL_KINDS=("labels")
                fi

                for LABEL_KIND in "${LABEL_KINDS[@]}"
                do
                    INPUT_FILE="$INVERSE_DIR/${BASE_TAG}_${LABEL_KIND}.fif"
                    if [ "$TASK" = "task" ] && [ ! -f "$INPUT_FILE" ]; then
                        ALT_INVERSE_DIR="$DERIV_ROOT/sub-$SUB/ses-$SES/source_reconstruction/inverse_solution/$MODE/$METHOD/task/cov$COV_TAG"
                        ALT_INPUT_FILE="$ALT_INVERSE_DIR/${BASE_TAG}_${LABEL_KIND}.fif"
                        if [ -f "$ALT_INPUT_FILE" ]; then
                            INVERSE_DIR="$ALT_INVERSE_DIR"
                            INPUT_FILE="$ALT_INPUT_FILE"
                        fi
                    fi
                    OUTPUT_DIR="$INVERSE_DIR/epochs"
                    BASELINE_FILE="$OUTPUT_DIR/${BASE_TAG}_${LABEL_KIND}_baseline-epo.fif"
                    NONBASELINE_FILE="$OUTPUT_DIR/${BASE_TAG}_${LABEL_KIND}_nonbaseline-epo.fif"

                    if [ ! -f "$INPUT_FILE" ]; then
                        echo "Skipping missing labels FIF: $INPUT_FILE"
                        continue
                    fi

                    if [ -f "$BASELINE_FILE" ] && [ -f "$NONBASELINE_FILE" ]; then
                        echo "Skipping already epoched: sub-$SUB ses-$SES task-$TASK mode-$MODE method-$METHOD cov-$PARAM_COV_LABEL $LABEL_KIND"
                    else
                        echo "$SUB $SES $TASK $MODE $METHOD $PARAM_COV_LABEL $LABEL_KIND $CONNECTIVITY_BASELINE $N_EPOCHS_PER_BLOCK" >> "$PARAM_LIST"
                    fi
                done
            done
        done
    done
done

NUM_FILES=$(wc -l < "$PARAM_LIST")

if [ "$NUM_FILES" -eq 0 ]; then
    echo "No new epoching jobs found."
    exit 0
fi

echo "Found $NUM_FILES epoching jobs to process."
echo "Submitting Job Array..."

sbatch --array=1-"$NUM_FILES" "$RUN_SCRIPT"
