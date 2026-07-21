#!/bin/bash
#Author Stavriani Skarvelaki

#This script is used to submit the array job for filtering and downsampling EEG data. It first scans the raw data directory for files that match the expected pattern, checks if the corresponding output file 
#already exists in the derivatives directory, and if not, adds the subject, session, and task information to a parameter list. Finally, it submits a SLURM array job to process all the files in the parameter list 
#using the specified run script. Change the filename pattern in line 20 and 22. If you leave the "task-"" part open it will processs all the BIDS files. The output files have permissions only for the user that run the scirpt,
#but the permissions can be changed in the next script for annotations. 

DATA_ROOT="/work/uphummel/studies/tTIS-EEG/data/raw/EEG"
DERIV_ROOT="/work/uphummel/studies/tTIS-EEG/derivatives/EEG"

LOG_DIR="/work/uphummel/studies/tTIS-EEG/code/Maelys/logs" #Specific log directory for this step .err are the errors and .out are the outputs of the scirpt, they are upfated live during the time that the job is running.

PARAM_LIST="/work/uphummel/studies/tTIS-EEG/code/Maelys/files_for_bem.txt"
RUN_SCRIPT="/home/clerget/Scripts/code/05_run_EEG_montage_BEM.sh"

mkdir -p "$LOG_DIR"

# Empty the parameter list before rebuilding it
> "$PARAM_LIST"

echo "Scanning $DATA_ROOT for task-RSpre EEG files..." # <--- change filename here !!

find "$DATA_ROOT" -type f -name "sub-*_ses-*_task-RSpre_eeg.vhdr" | sort | while read -r FILE # <--- and here !! 

do
    BASENAME=$(basename "$FILE")

    # Extract subject and session from filename
    SUB=$(echo "$BASENAME" | sed -E 's/^sub-([^_]+)_ses-.*/\1/')
    SES=$(echo "$BASENAME" | sed -E 's/^sub-[^_]+_ses-([^_]+)_task-.*/\1/')
    TASK=$(echo "$BASENAME" | sed -E 's/^sub-[^_]+_ses-[^_]+_task-([^_]+)_eeg\.vhdr$/\1/')

    # Expected output file
    OUTPUT_FILE="$DERIV_ROOT/sub-$SUB/ses-$SES/source_reconstruction/personalized_montage/${SUB}_ses${SES}_task-${TASK}_personalized_montage.fif"

    ## Skip already processed files
    #if [ -f "$OUTPUT_FILE" ]; then
    #    echo "Skipping already processed: sub-$SUB ses-$SES task-$TASK"
    #else
    #    echo "$SUB $SES $TASK" >> "$PARAM_LIST" # if you want to process already existing files, keep this line only and change python script
    #fi

    echo "$SUB $SES $TASK" >> "$PARAM_LIST"
done

# Count how many jobs we found
NUM_FILES=$(wc -l < "$PARAM_LIST")

if [ "$NUM_FILES" -eq 0 ]; then
    echo "No new files found to process."
    exit 0
fi

echo "Found $NUM_FILES jobs to process."
echo "Submitting Job Array..."

# Submit the array
sbatch --array=1-"$NUM_FILES" "$RUN_SCRIPT"
