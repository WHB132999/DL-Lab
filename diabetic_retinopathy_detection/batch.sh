#!/bin/bash -l

# Slurm parameters
#SBATCH --job-name=group-15
#SBATCH --output=group_15-%j.%N.out
#SBATCH --time=1-00:00:00
#SBATCH --gpus=1

# Activate everything you need
module load cuda/11.2
# Run your python code
python3 main_test.py
