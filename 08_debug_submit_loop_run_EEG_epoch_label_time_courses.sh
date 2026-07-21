#!/bin/bash
# Author Stavriani Skarvelaki / Maelys Clerget

# Debug submitter for a small set of ROI label epoching jobs.

LOG_DIR="/work/uphummel/studies/tTIS-EEG/code/Maelys/logs"
PARAM_LIST="/work/uphummel/studies/tTIS-EEG/code/Maelys/files_for_epoch_label_time_courses.txt"
RUN_SCRIPT="/home/clerget/Scripts/code/08_run_EEG_epoch_label_time_courses.sh"

mkdir -p "$LOG_DIR"
> "$PARAM_LIST"

# ---- ONE TEST SUBJECT ----
SUB="41Y01"
SES="1"
TASK="RSstim"
MODES=("surface" "volume" "mixed")
METHOD="MNE"
COV_LABEL="15"
CONNECTIVITY_BASELINE="stim"

for MODE in "${MODES[@]}"
do
    if [ "$MODE" = "mixed" ]; then
        LABEL_KINDS=("surface_labels" "volume_labels")
    else
        LABEL_KINDS=("labels")
    fi

    for LABEL_KIND in "${LABEL_KINDS[@]}"
    do
        echo "$SUB $SES $TASK $MODE $METHOD $COV_LABEL $LABEL_KIND $CONNECTIVITY_BASELINE" >> "$PARAM_LIST"
    done
done

echo "Added test ROI label epoching job(s):"
cat "$PARAM_LIST"

NUM_FILES=$(wc -l < "$PARAM_LIST")

if [ "$NUM_FILES" -eq 0 ]; then
    echo "No jobs found."
    exit 0
fi

echo "Submitting $NUM_FILES epoching job(s)."

sbatch --array=1-"$NUM_FILES" "$RUN_SCRIPT"
