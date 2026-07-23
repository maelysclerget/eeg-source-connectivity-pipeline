#!/bin/bash
# Author Stavriani Skarvelaki / Maelys Clerget

#This script is used to submit the array job for filtering and downsampling EEG data. It first scans the raw data directory for files that match the expected pattern, checks if the corresponding output file 
#already exists in the derivatives directory, and if not, adds the subject, session, and task information to a parameter list. Finally, it submits a SLURM array job to process all the files in the parameter list 
#using the specified run script. Change the filename pattern in line 20 and 22. If you leave the "task-"" part open it will processs all the BIDS files. The output files have permissions only for the user that run the scirpt,
#but the permissions can be changed in the next script for annotations. 

DERIV_ROOT="/work/uphummel/studies/tTIS-EEG/derivatives/EEG"

LOG_DIR="/work/uphummel/studies/tTIS-EEG/code/Maelys/logs"
PARAM_LIST="/work/uphummel/studies/tTIS-EEG/code/Maelys/files_for_inverse_solution_only.txt"
RUN_SCRIPT="/home/clerget/Scripts/code/06_run_EEG_inverse_solution_only.sh"

# Source space modes to run.
MODES=("surface" "volume" "mixed")

# Minimum-norm method choices: MNE, sLORETA, eLORETA.
METHODS=("MNE" "sLORETA" "eLORETA")

# Use 20-second windows for the noise covariance.
COV_DURATION="20.0"

mkdir -p "$LOG_DIR"
> "$PARAM_LIST"

echo "Scanning $DERIV_ROOT for task-RSpre non-interpolated preprocessed EEG FIF files."

find "$DERIV_ROOT" -type f -name "sub-*_ses-*_task-RSpre_eeg_not_interpolated_final_preprocessed_eeg.fif" | sort | while read -r FILE
do
    BASENAME=$(basename "$FILE")

    SUB=$(echo "$BASENAME" | sed -E 's/^sub-([^_]+)_ses-.*/\1/')
    SES=$(echo "$BASENAME" | sed -E 's/^sub-[^_]+_ses-([^_]+)_task-.*/\1/')
    TASK=$(echo "$BASENAME" | sed -E 's/^sub-[^_]+_ses-[^_]+_task-([^_]+)_eeg_not_interpolated_final_preprocessed_eeg\.fif$/\1/')

    if [ "$TASK" != "RSpre" ]; then
        continue
    fi

    INPUT_FILE="$DERIV_ROOT/sub-$SUB/ses-$SES/personalized_montage/${SUB}_ses${SES}_personalized_montage.fif"
    if [ ! -f "$INPUT_FILE" ]; then
        echo "Skipping missing personalized montage: sub-$SUB ses-$SES task-$TASK"
    else
        for MODE in "${MODES[@]}"
        do
            FWD_FILE="$DERIV_ROOT/sub-$SUB/ses-$SES/source_reconstruction/forward_solution/$MODE/${SUB}_ses${SES}_src-${MODE}-fwd.fif"
            if [ ! -f "$FWD_FILE" ]; then
                echo "Skipping missing forward solution: sub-$SUB ses-$SES mode-$MODE"
                continue
            fi

            for METHOD in "${METHODS[@]}"
            do
                COV_LABELS=("RSpre")
                if [ "$TASK" = "task" ]; then
                    COV_LABELS=("s4" "s15")
                fi

                ALL_DONE=1
                for COV_LABEL in "${COV_LABELS[@]}"
                do
                    if [ "$TASK" = "task" ]; then
                        OUTPUT_DIR="$DERIV_ROOT/sub-$SUB/ses-$SES/$TASK/inverse_solution/$MODE/$METHOD/cov$COV_LABEL"
                    else
                        OUTPUT_DIR="$DERIV_ROOT/sub-$SUB/ses-$SES/$TASK/inverse_solution/$MODE/$METHOD"
                    fi

                    BASE_TAG="${SUB}_ses${SES}_task-${TASK}_src-${MODE}_method-${METHOD}"
                    if [ "$TASK" = "task" ]; then
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
