#!/bin/bash
#Author Stavriani Skarvelaki

#SBATCH --job-name=eeg_montage_bem
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --mem=64G 
#SBATCH --cpus-per-task=15
#SBATCH --error=/home/clerget/Scripts/preprocessing_pipeline/logs/%x_%A_%a.err
#SBATCH --output=/home/clerget/Scripts/preprocessing_pipeline/logs/%x_%A_%a.out


# SS: Important to check memory depending on the size of the data, for 50kHz sampling frequency, 15min recording --> 28G and cpus=5, 30min recording --> 64G and cpus=15
#Always follow the rule: requested memory <= requested CPUs × 7000 MB

# 1. Set default permissions for created files to be group-readable and writable
umask 0002 

# 2. Load Modules
module purge
module load gcc python

cd /work/uphummel/studies/tTIS-EEG/code/Maelys/

# 3. Activate Environment
#source /work/uphummel/shared/software/python_envs/mne_venv/bin/activate

# 4. DeteMODULE rmine which file to process
# $SLURM_ARRAY_TASK_ID is the index (1, 2, 3...)
# sed -n "${p}p" prints just that line from the file list
PARAM_LIST="/work/uphummel/studies/tTIS-EEG/code/Maelys/files_for_bem.txt"
CURRENT_FILE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$PARAM_LIST")

# 5. Check if we got a file
if [ -z "$CURRENT_FILE" ]; then
    echo "Error: No file found for Array ID $SLURM_ARRAY_TASK_ID"
    exit 1
fi

read -r SUB SES TASK <<< "$CURRENT_FILE"

echo "SUB = '$SUB'"
echo "SES = '$SES'"
echo "TASK = '$TASK'"

echo "Processing sub-$SUB ses-$SES task-$TASK"

# 6. Run Python Script

MNE_FS_IMG="/work/uphummel/shared/software/containers/mne_freesurfer/mne_freesurfer_8.1.sif"
FS_LICENSE="/work/uphummel/shared/software/containers/license.txt"
SUBJECTS_DIR="/work/uphummel/studies/tTIS-EEG/derivatives/MRI/freesurfer"   

apptainer exec \
    --bind /work/uphummel:/work/uphummel \
    --bind /home/skarvela:/home/skarvela \
    --env FS_LICENSE="$FS_LICENSE" \
    --env SUBJECTS_DIR="$SUBJECTS_DIR" \
    "$MNE_FS_IMG" \
    bash -lc '
        export FREESURFER_HOME=/usr/local/freesurfer/8.1.0-1

        cd /home/clerget/Scripts/code

        echo "FREESURFER_HOME=$FREESURFER_HOME"
        echo "SUBJECTS_DIR=$SUBJECTS_DIR"
        command -v mri_watershed

        python 05_EEG_montage_BEM.py \
            --subject "'"$SUB"'" \
            --session "'"$SES"'" \
            --task "'"$TASK"'"
    '
