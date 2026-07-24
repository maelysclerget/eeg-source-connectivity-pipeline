#!/bin/bash
# Author Stavriani Skarvelaki / Maelys Clerget

# Submit ROI label epoching jobs as a SLURM array.
# This scans raw EEG files, checks for step 06 label time-course FIF outputs,
# skips already-created step 08 Epochs FIF files, and submits one array row per
# missing subject/session/task/mode/method/covariance/label-kind combination.

DERIV_ROOT="/work/uphummel/studies/tTIS-EEG/derivatives/EEG"

LOG_DIR="/work/uphummel/studies/tTIS-EEG/logs"
PARAM_LIST="/work/uphummel/studies/tTIS-EEG/code/Maelys/files_for_epoch_label_time_courses.txt"
RUN_SCRIPT="/home/clerget/Scripts/code/08_run_EEG_epoch_label_time_courses.sh"

# Source space modes and inverse methods to epoch.
MODES=("surface" "volume" "mixed")
METHODS=("MNE" "sLORETA" "eLORETA")

# For task data, epoch one labels output; the covariance tag is not kept in
# the epoch output filename.
TASK_COV_LABELS=("15")

mkdir -p "$LOG_DIR"
> "$PARAM_LIST"

echo "Scanning $DERIV_ROOT for task-RSpre non-interpolated preprocessed EEG FIF files."

find "$DERIV_ROOT" -type f -name "sub-*_ses-*_task-RSpre_eeg_not_interpolated_final_preprocessed_eeg.fif" | sort | while read -r FILE
do
    BASENAME=$(basename "$FILE")

    SUB=$(echo "$BASENAME" | sed -E 's/^sub-([^_]+)_ses-.*/\1/')
    SES=$(echo "$BASENAME" | sed -E 's/^sub-[^_]+_ses-([^_]+)_task-.*/\1/')
    TASK=$(echo "$BASENAME" | sed -E 's/^sub-[^_]+_ses-[^_]+_task-([^_]+)_eeg_not_interpolated_final_preprocessed_eeg\.fif$/\1/')

    if [ "$TASK" != "RSpre" ]; then
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
                    INVERSE_DIR="$DERIV_ROOT/sub-$SUB/ses-$SES/$TASK/inverse_solution/$MODE/$METHOD/cov$COV_TAG"
                    BASE_TAG="${SUB}_ses${SES}_task-${TASK}_src-${MODE}_method-${METHOD}_cov-${COV_TAG}"
                    PARAM_COV_LABEL="$COV_LABEL"
                else
                    INVERSE_DIR="$DERIV_ROOT/sub-$SUB/ses-$SES/$TASK/inverse_solution/$MODE/$METHOD"
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
                    OUTPUT_DIR="$DERIV_ROOT/sub-$SUB/ses-$SES/$TASK/epochs/$MODE/$METHOD"
                    OUTPUT_BASE_TAG="${SUB}_ses${SES}_task-${TASK}_src-${MODE}_method-${METHOD}"
                    if [ "$TASK" = "task" ]; then
                        OUTPUT_FILE="$OUTPUT_DIR/${OUTPUT_BASE_TAG}_${LABEL_KIND}_task_fixed-epo.fif"
                    else
                        OUTPUT_FILE="$OUTPUT_DIR/${OUTPUT_BASE_TAG}_${LABEL_KIND}_rs_s15_epochs-epo.fif"
                    fi

                    if [ ! -f "$INPUT_FILE" ]; then
                        echo "Skipping missing labels FIF: $INPUT_FILE"
                        continue
                    fi

                    if [ -f "$OUTPUT_FILE" ]; then
                        echo "Skipping already epoched: sub-$SUB ses-$SES task-$TASK mode-$MODE method-$METHOD cov-$PARAM_COV_LABEL $LABEL_KIND"
                    else
                        echo "$SUB $SES $TASK $MODE $METHOD $PARAM_COV_LABEL $LABEL_KIND" >> "$PARAM_LIST"
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
