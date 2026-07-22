#!/bin/bash
# Author Stavriani Skarvelaki / Maelys Clerget

LOG_DIR="/work/uphummel/studies/tTIS-EEG/code/Maelys/logs"
PARAM_LIST="/work/uphummel/studies/tTIS-EEG/code/Maelys/files_for_connectivity.txt"
RUN_SCRIPT="/home/clerget/Scripts/code/07_run_EEG_connectivity.sh"

mkdir -p "$LOG_DIR"
> "$PARAM_LIST"

# ---- ONE TEST SUBJECT ----
SUB="41Y01"
SES="1"
TASK="task"
MODES=("surface" "volume" "mixed")
METHOD="MNE"
BASELINE_KINDS=("baseline_stim")
# To test both task baselines, use:
# BASELINE_KINDS=("baseline_stim" "baseline_no_stim")

for MODE in "${MODES[@]}"; do
    if [ "$TASK" = "task" ]; then
        for BASELINE_KIND in "${BASELINE_KINDS[@]}"; do
            echo "$SUB $SES $TASK $MODE $METHOD $BASELINE_KIND" >> "$PARAM_LIST"
        done
    else
        echo "$SUB $SES $TASK $MODE $METHOD baseline_stim" >> "$PARAM_LIST"
    fi
done

echo "Added test source-connectivity job:"
cat "$PARAM_LIST"

NUM_FILES=$(wc -l < "$PARAM_LIST")

if [ "$NUM_FILES" -eq 0 ]; then
    echo "No jobs found."
    exit 0
fi

echo "Submitting $NUM_FILES connectivity job(s)."

sbatch --array=1-"$NUM_FILES" "$RUN_SCRIPT"
