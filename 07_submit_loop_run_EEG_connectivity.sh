#!/bin/bash
# Author Stavriani Skarvelaki / Maelys Clerget

# Submit source-level EEG connectivity jobs as a SLURM array.
# The script scans step 08 ROI epoch outputs, checks whether the source
# connectivity summary already exists, and submits one job per missing result.

DERIV_ROOT="/work/uphummel/studies/tTIS-EEG/derivatives/EEG"

LOG_DIR="/work/uphummel/studies/tTIS-EEG/code/Maelys/logs"
PARAM_LIST="/work/uphummel/studies/tTIS-EEG/code/Maelys/files_for_connectivity.txt"
RUN_SCRIPT="/home/clerget/Scripts/code/07_run_EEG_connectivity.sh"

# Choose source space and inverse methods here.
MODES=("surface" "volume" "mixed")
METHODS=("MNE" "sLORETA" "eLORETA")
BASELINE_KINDS=("baseline_stim")
# To run both task baselines, use:
# BASELINE_KINDS=("baseline_stim" "baseline_no_stim")

mkdir -p "$LOG_DIR"
> "$PARAM_LIST"

echo "Scanning ROI epoch outputs in $DERIV_ROOT."

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
            EPOCH_DIR="$DERIV_ROOT/sub-$SUB/ses-$SES/$TASK/epochs/$MODE/$METHOD"
            CONNECTIVITY_DIR="$DERIV_ROOT/sub-$SUB/ses-$SES/$TASK/connectivity/$MODE/$METHOD"
            BASE_TAG="${SUB}_ses${SES}_task-${TASK}_src-${MODE}_method-${METHOD}"

            if [ "$TASK" = "task" ]; then
                for BASELINE_KIND in "${BASELINE_KINDS[@]}"
                do
                    BASELINE_LABEL=${BASELINE_KIND#baseline_}
                    OUTPUT_FILE="$CONNECTIVITY_DIR/${BASE_TAG}_baseline-${BASELINE_LABEL}_connectivity_summary.csv"
                    if [ "$MODE" = "mixed" ]; then
                        INPUT_FILE="$EPOCH_DIR/${BASE_TAG}_surface_labels_task_fixed-epo.fif"
                        INPUT_FILE_2="$EPOCH_DIR/${BASE_TAG}_volume_labels_task_fixed-epo.fif"
                        BASELINE_FILE="$EPOCH_DIR/${BASE_TAG}_surface_labels_${BASELINE_KIND}-epo.fif"
                        BASELINE_FILE_2="$EPOCH_DIR/${BASE_TAG}_volume_labels_${BASELINE_KIND}-epo.fif"
                    else
                        INPUT_FILE="$EPOCH_DIR/${BASE_TAG}_labels_task_fixed-epo.fif"
                        INPUT_FILE_2=""
                        BASELINE_FILE="$EPOCH_DIR/${BASE_TAG}_labels_${BASELINE_KIND}-epo.fif"
                        BASELINE_FILE_2=""
                    fi

                    if [ ! -f "$INPUT_FILE" ] || \
                       { [ -n "$INPUT_FILE_2" ] && [ ! -f "$INPUT_FILE_2" ]; } || \
                       { [ -n "$BASELINE_FILE" ] && [ ! -f "$BASELINE_FILE" ]; } || \
                       { [ -n "$BASELINE_FILE_2" ] && [ ! -f "$BASELINE_FILE_2" ]; }; then
                        echo "Skipping missing ROI epochs: sub-$SUB ses-$SES task-$TASK mode-$MODE method-$METHOD baseline-$BASELINE_KIND"
                        continue
                    fi

                    if [ -f "$OUTPUT_FILE" ]; then
                        echo "Skipping already processed: sub-$SUB ses-$SES task-$TASK mode-$MODE method-$METHOD baseline-$BASELINE_KIND"
                    else
                        echo "$SUB $SES $TASK $MODE $METHOD $BASELINE_KIND" >> "$PARAM_LIST"
                    fi
                done
            else
                OUTPUT_FILE="$CONNECTIVITY_DIR/${BASE_TAG}_connectivity_summary.csv"
                if [ "$MODE" = "mixed" ]; then
                    INPUT_FILE="$EPOCH_DIR/${BASE_TAG}_surface_labels_rs_s15_epochs-epo.fif"
                    INPUT_FILE_2="$EPOCH_DIR/${BASE_TAG}_volume_labels_rs_s15_epochs-epo.fif"
                else
                    INPUT_FILE="$EPOCH_DIR/${BASE_TAG}_labels_rs_s15_epochs-epo.fif"
                    INPUT_FILE_2=""
                fi

                if [ ! -f "$INPUT_FILE" ] || { [ -n "$INPUT_FILE_2" ] && [ ! -f "$INPUT_FILE_2" ]; }; then
                    echo "Skipping missing ROI epochs: sub-$SUB ses-$SES task-$TASK mode-$MODE method-$METHOD"
                    continue
                fi

                if [ -f "$OUTPUT_FILE" ]; then
                    echo "Skipping already processed: sub-$SUB ses-$SES task-$TASK mode-$MODE method-$METHOD"
                else
                    echo "$SUB $SES $TASK $MODE $METHOD baseline_stim" >> "$PARAM_LIST"
                fi
            fi
        done
    done
done

NUM_FILES=$(wc -l < "$PARAM_LIST")

if [ "$NUM_FILES" -eq 0 ]; then
    echo "No new connectivity jobs found."
    exit 0
fi

echo "Found $NUM_FILES connectivity jobs to process."
echo "Submitting Job Array..."

sbatch --array=1-"$NUM_FILES" "$RUN_SCRIPT"
