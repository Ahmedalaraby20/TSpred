#!/bin/bash
#SBATCH --job-name=TSpred
#SBATCH --partition=leinegpu_long
#SBATCH --gres=gpu:1                # Request 1 GPU per task
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=8000M
#SBATCH --array=0-99                # Create a job array with 100 tasks
#SBATCH --time=60:00:00

# Load Conda environment
source activate TCR

# Define the input files
input_files=($(ls h2kbinputs/*.csv))

# Calculate the total number of files and split them into 100 parts
total_files=${#input_files[@]}
split_size=$(( (total_files + 99) / 100 ))  # Split into 100 parts

# Get the subset of files for this array task
start=$(( SLURM_ARRAY_TASK_ID * split_size ))
end=$(( (SLURM_ARRAY_TASK_ID + 1) * split_size ))
subset=("${input_files[@]:$start:$split_size}")

# Execute the Python script with the subset of files
python predicth2kb.py "${subset[@]}"

