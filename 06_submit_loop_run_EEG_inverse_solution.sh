#!/bin/bash
# Author Stavriani Skarvelaki / Maelys Clerget

#This script is used to submit the array job for filtering and downsampling EEG data. It first scans the raw data directory for files that match the expected pattern, checks if the corresponding output file 
#already exists in the derivatives directory, and if not, adds the subject, session, and task information to a parameter list. Finally, it submits a SLURM array job to process all the files in the parameter list 
#using the specified run script. Change the filename pattern in line 20 and 22. If you leave the "task-"" part open it will processs all the BIDS files. The output files have permissions only for the user that run the scirpt,
#but the permissions can be changed in the next script for annotations. 

DATA_ROOT="/work/uphummel/studies/tTIS-EEG/data/raw/EEG"
DERIV_ROOT="/work/uphummel/studies/tTIS-EEG/derivatives/EEG"

LOG_DIR="/work/uphummel/studies/tTIS-EEG/code/Maelys/logs"
PARAM_LIST="/work/uphummel/studies/tTIS-EEG/code/Maelys/files_for_inverse_solution.txt"
RUN_SCRIPT="/home/clerget/Scripts/code/06_run_EEG_inverse_solution.sh"

# Source space modes to run.
MODES=("surface" "volume" "mixed")

# Minimum-norm method choices: MNE, sLORETA, eLORETA.
METHODS=("MNE" "sLORETA" "eLORETA")

# Use 20-second windows for the noise covariance.
COV_DURATION="20.0"

mkdir -p "$LOG_DIR"
> "$PARAM_LIST"

echo "Scanning $DATA_ROOT for task-task and task-RSpre EEG files."

find "$DATA_ROOT" -type f -name "sub-*_ses-*_task-*_eeg.vhdr" | sort | while read -r FILE
do
    BASENAME=$(basename "$FILE")

    SUB=$(echo "$BASENAME" | sed -E 's/^sub-([^_]+)_ses-.*/\1/')
    SES=$(echo "$BASENAME" | sed -E 's/^sub-[^_]+_ses-([^_]+)_task-.*/\1/')
    TASK=$(echo "$BASENAME" | sed -E 's/^sub-[^_]+_ses-[^_]+_task-([^_]+)_eeg\.vhdr$/\1/')

    if [ "$TASK" != "task" ] && [ "$TASK" != "RSpre" ]; then
        continue
    fi

    INPUT_FILE="$DERIV_ROOT/sub-$SUB/ses-$SES/source_reconstruction/personalized_montage/${SUB}_ses${SES}_personalized_montage.fif"
    if [ ! -f "$INPUT_FILE" ]; then
        echo "Skipping missing personalized montage: sub-$SUB ses-$SES task-$TASK"
    else
        for MODE in "${MODES[@]}"
        do
            for METHOD in "${METHODS[@]}"
            do
                COV_LABELS=("RSpre")
                if [ "$TASK" = "task" ]; then
                    COV_LABELS=("4" "15")
                fi

                ALL_DONE=1
                for COV_LABEL in "${COV_LABELS[@]}"
                do
                    if [ "$TASK" = "task" ]; then
                        OUTPUT_DIR="$DERIV_ROOT/sub-$SUB/ses-$SES/source_reconstruction/inverse_solution/$MODE/$METHOD/task/cov$COV_LABEL"
                    elif [ "$TASK" = "RSpre" ]; then
                        OUTPUT_DIR="$DERIV_ROOT/sub-$SUB/ses-$SES/source_reconstruction/inverse_solution/$MODE/$METHOD/RSpre"
                    else
                        OUTPUT_DIR="$DERIV_ROOT/sub-$SUB/ses-$SES/source_reconstruction/inverse_solution/$MODE/$METHOD/task-$TASK/cov-$COV_LABEL"
                    fi

                    BASE_TAG="${SUB}_ses${SES}_task-${TASK}_src-${MODE}_method-${METHOD}"
                    if [ "$TASK" != "RSpre" ]; then
                        BASE_TAG="${BASE_TAG}_cov-${COV_LABEL}"
                    fi

                    if [ "$MODE" = "surface" ]; then
                        OUTPUT_FILE="$OUTPUT_DIR/${BASE_TAG}_morph-fsaverage_stc-lh.stc"
                    elif [ "$MODE" = "mixed" ]; then
                        OUTPUT_FILE="$OUTPUT_DIR/${BASE_TAG}_surface_labels.fif"
                    else
                        OUTPUT_FILE="$OUTPUT_DIR/${BASE_TAG}_labels.fif"
                    fi

                    if [ ! -f "$OUTPUT_FILE" ]; then
                        ALL_DONE=0
                    fi
                done

                if [ "$ALL_DONE" -eq 1 ]; then
                    echo "Skipping already processed: sub-$SUB ses-$SES task-$TASK mode-$MODE method-$METHOD"
                else
                    MORPH_TO=""
                    if [ "$MODE" = "surface" ]; then
                        MORPH_TO="fsaverage"
                    fi

                    echo "$SUB $SES $TASK $MODE $METHOD $COV_DURATION $MORPH_TO" >> "$PARAM_LIST"
                fi
            done
        done
    fi
done

NUM_FILES=$(wc -l < "$PARAM_LIST")

if [ "$NUM_FILES" -eq 0 ]; then
    echo "No new files found to process."
    exit 0
fi

echo "Found $NUM_FILES jobs to process."
echo "Submitting Job Array..."

sbatch --array=1-"$NUM_FILES" "$RUN_SCRIPT"
