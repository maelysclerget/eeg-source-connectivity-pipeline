#!/bin/bash
# Author Stavriani Skarvelaki / Maelys Clerget

# Debug submitter for a single inverse-solution job.

DATA_ROOT="/work/uphummel/studies/tTIS-EEG/data/raw/EEG"
DERIV_ROOT="/work/uphummel/studies/tTIS-EEG/derivatives/EEG"

LOG_DIR="/work/uphummel/studies/tTIS-EEG/code/Maelys/logs"
PARAM_LIST="/work/uphummel/studies/tTIS-EEG/code/Maelys/files_for_inverse_solution.txt"
RUN_SCRIPT="/home/clerget/Scripts/code/06_run_EEG_inverse_solution.sh"

mkdir -p "$LOG_DIR"
> "$PARAM_LIST"

# ---- ONE TEST SUBJECT ----
SUB="41Y01"
SES="1"
TASK="RSstim"
MODES=("surface" "volume" "mixed")
METHOD="MNE"
COV_DURATION="20.0"

for MODE in "${MODES[@]}"
do
    echo "$SUB $SES $TASK $MODE $METHOD $COV_DURATION fsaverage" >> "$PARAM_LIST"
done

echo "Added test inverse-solution subject:"
cat "$PARAM_LIST"

NUM_FILES=$(wc -l < "$PARAM_LIST")

if [ "$NUM_FILES" -eq 0 ]; then
    echo "No jobs found."
    exit 0
fi

echo "Submitting $NUM_FILES job."

sbatch --array=1-"$NUM_FILES" "$RUN_SCRIPT"
