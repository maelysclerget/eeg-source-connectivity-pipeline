#!/bin/bash
# Author Stavriani Skarvelaki / Maelys Clerget

# Submit RSpre source-level EEG connectivity jobs as a SLURM array.
# The script scans step 06 RSpre ROI outputs, checks whether the source
# connectivity summary already exists, and submits one job per missing result.

DERIV_ROOT="/work/uphummel/studies/tTIS-EEG/derivatives/EEG"
RAW_ROOT="/work/uphummel/studies/tTIS-EEG/data/raw/EEG"

LOG_DIR="/work/uphummel/studies/tTIS-EEG/code/Maelys/logs"
PARAM_LIST="/work/uphummel/studies/tTIS-EEG/code/Maelys/files_for_connectivity.txt"
RUN_SCRIPT="/home/clerget/Scripts/code/07_run_EEG_connectivity.sh"

# Choose source space and inverse methods here.
MODES=("surface" "volume" "mixed")
METHODS=("MNE" "sLORETA" "eLORETA")

mkdir -p "$LOG_DIR"
> "$PARAM_LIST"

echo "Scanning RSpre inverse-solution outputs in $DERIV_ROOT."

find "$RAW_ROOT" -type f -name "sub-*_ses-*_task-RSpre_eeg.vhdr" | sort | while read -r FILE
do
    BASENAME=$(basename "$FILE")

    SUB=$(echo "$BASENAME" | sed -E 's/^sub-([^_]+)_ses-.*/\1/')
    SES=$(echo "$BASENAME" | sed -E 's/^sub-[^_]+_ses-([^_]+)_task-.*/\1/')
    TASK=$(echo "$BASENAME" | sed -E 's/^sub-[^_]+_ses-[^_]+_task-([^_]+)_eeg\.vhdr$/\1/')

    if [ "$TASK" != "RSpre" ]; then
        continue
    fi

    for MODE in "${MODES[@]}"
    do
        for METHOD in "${METHODS[@]}"
        do
            INVERSE_DIR="$DERIV_ROOT/sub-$SUB/ses-$SES/source_reconstruction/inverse_solution/$MODE/$METHOD/RSpre"
            CONNECTIVITY_DIR="$DERIV_ROOT/sub-$SUB/ses-$SES/source_reconstruction/connectivity/$MODE/$METHOD/RSpre"
            BASE_TAG="${SUB}_ses${SES}_task-${TASK}_src-${MODE}_method-${METHOD}"

            if [ "$MODE" = "mixed" ]; then
                INPUT_FILE="$INVERSE_DIR/${BASE_TAG}_surface_labels.fif"
                INPUT_FILE_2="$INVERSE_DIR/${BASE_TAG}_volume_labels.fif"
            else
                INPUT_FILE="$INVERSE_DIR/${BASE_TAG}_labels.fif"
                INPUT_FILE_2=""
            fi

            if [ ! -f "$INPUT_FILE" ] || { [ -n "$INPUT_FILE_2" ] && [ ! -f "$INPUT_FILE_2" ]; }; then
                echo "Skipping missing RSpre ROI time course: sub-$SUB ses-$SES mode-$MODE method-$METHOD"
                continue
            fi

            OUTPUT_FILE="$CONNECTIVITY_DIR/${BASE_TAG}_connectivity_summary.csv"

            if [ -f "$OUTPUT_FILE" ]; then
                echo "Skipping already processed: sub-$SUB ses-$SES task-$TASK mode-$MODE method-$METHOD"
            else
                echo "$SUB $SES $TASK $MODE $METHOD" >> "$PARAM_LIST"
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
