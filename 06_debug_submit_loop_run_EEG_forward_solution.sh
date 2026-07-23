#!/bin/bash
# Author Stavriani Skarvelaki / Maelys Clerget

# Debug submitter for one subject/session forward-solution job set.

DATA_ROOT="/work/uphummel/studies/tTIS-EEG/data/raw/EEG"
DERIV_ROOT="/work/uphummel/studies/tTIS-EEG/derivatives/EEG"

LOG_DIR="/work/uphummel/studies/tTIS-EEG/code/Maelys/logs"
PARAM_LIST="/work/uphummel/studies/tTIS-EEG/code/Maelys/files_for_forward_solution.txt"
RUN_SCRIPT="/home/clerget/Scripts/code/06_run_EEG_forward_solution.sh"

mkdir -p "$LOG_DIR"
> "$PARAM_LIST"

# ---- ONE TEST SUBJECT ----
SUB="41Y01"
SES="1"
MODES=("surface" "volume" "mixed")

for MODE in "${MODES[@]}"
do
    echo "$SUB $SES $MODE" >> "$PARAM_LIST"
done

echo "Added test forward-solution subject:"
cat "$PARAM_LIST"

NUM_FILES=$(wc -l < "$PARAM_LIST")

if [ "$NUM_FILES" -eq 0 ]; then
    echo "No jobs found."
    exit 0
fi

echo "Submitting $NUM_FILES job."

sbatch --array=1-"$NUM_FILES" "$RUN_SCRIPT"
