#!/bin/bash
#Author Stavriani Skarvelaki

#This script is used to submit the array job for filtering and downsampling EEG data. It first scans the raw data directory for files that match the expected pattern, checks if the corresponding output file 
#already exists in the derivatives directory, and if not, adds the subject, session, and task information to a parameter list. Finally, it submits a SLURM array job to process all the files in the parameter list 
#using the specified run script. Change the filename pattern in line 20 and 22. If you leave the "task-"" part open it will processs all the BIDS files. The output files have permissions only for the user that run the scirpt,
#but the permissions can be changed in the next script for annotations. 

#!/bin/bash
#Author Stavriani Skarvelaki

DATA_ROOT="/work/uphummel/studies/tTIS-EEG/data/raw/EEG"
DERIV_ROOT="/work/uphummel/studies/tTIS-EEG/derivatives/EEG"

LOG_DIR="/work/uphummel/studies/tTIS-EEG/code/Maelys/logs"

PARAM_LIST="/work/uphummel/studies/tTIS-EEG/code/Maelys/files_for_bem.txt"
RUN_SCRIPT="/home/clerget/Scripts/code/05_run_EEG_montage_BEM.sh"

mkdir -p "$LOG_DIR"

# Empty parameter list
> "$PARAM_LIST"

# ---- ONE TEST SUBJECT ----
SUB="41Y01"
SES="1"
TASK="task"

echo "$SUB $SES $TASK" >> "$PARAM_LIST"

echo "Added test subject:"
cat "$PARAM_LIST"

# Count jobs
NUM_FILES=$(wc -l < "$PARAM_LIST")

if [ "$NUM_FILES" -eq 0 ]; then
    echo "No jobs found."
    exit 0
fi

echo "Submitting $NUM_FILES job..."

sbatch --array=1-"$NUM_FILES" "$RUN_SCRIPT"
