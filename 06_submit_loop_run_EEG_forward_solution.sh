#!/bin/bash
# Author Stavriani Skarvelaki / Maelys Clerget

DERIV_ROOT="/work/uphummel/studies/tTIS-EEG/derivatives/EEG"

LOG_DIR="/work/uphummel/studies/tTIS-EEG/code/Maelys/logs"
PARAM_LIST="/work/uphummel/studies/tTIS-EEG/code/Maelys/files_for_forward_solution.txt"
RUN_SCRIPT="/home/clerget/Scripts/code/06_run_EEG_forward_solution.sh"

MODES=("surface" "volume" "mixed")

mkdir -p "$LOG_DIR"
> "$PARAM_LIST"

echo "Scanning $DERIV_ROOT for unique subject/session non-interpolated preprocessed EEG FIF files."

find "$DERIV_ROOT" -type f -name "sub-*_ses-*_task-RSpre_eeg_not_interpolated_final_preprocessed_eeg.fif" | sort | while read -r FILE
do
    BASENAME=$(basename "$FILE")

    SUB=$(echo "$BASENAME" | sed -E 's/^sub-([^_]+)_ses-.*/\1/')
    SES=$(echo "$BASENAME" | sed -E 's/^sub-[^_]+_ses-([^_]+)_task-.*/\1/')

    INPUT_FILE="$DERIV_ROOT/sub-$SUB/ses-$SES/personalized_montage/${SUB}_ses${SES}_personalized_montage.fif"
    TRANS_FILE="$DERIV_ROOT/sub-$SUB/ses-$SES/source_reconstruction/${SUB}_ses${SES}_trans.fif"

    if [ ! -f "$INPUT_FILE" ]; then
        echo "Skipping missing personalized montage: sub-$SUB ses-$SES"
        continue
    fi

    if [ ! -f "$TRANS_FILE" ]; then
        echo "Skipping missing trans file: sub-$SUB ses-$SES"
        continue
    fi

    for MODE in "${MODES[@]}"
    do
        OUTPUT_FILE="$DERIV_ROOT/sub-$SUB/ses-$SES/source_reconstruction/forward_solution/$MODE/${SUB}_ses${SES}_src-${MODE}-fwd.fif"
        if [ -f "$OUTPUT_FILE" ]; then
            echo "Skipping existing forward solution: sub-$SUB ses-$SES mode-$MODE"
        else
            JOB_LINE="$SUB $SES $MODE"
            if ! grep -qxF "$JOB_LINE" "$PARAM_LIST"; then
                echo "$JOB_LINE" >> "$PARAM_LIST"
            fi
        fi
    done
done

NUM_FILES=$(wc -l < "$PARAM_LIST")

if [ "$NUM_FILES" -eq 0 ]; then
    echo "No new forward solutions found to process."
    exit 0
fi

echo "Found $NUM_FILES jobs to process."
echo "Submitting Job Array..."

sbatch --array=1-"$NUM_FILES" "$RUN_SCRIPT"
