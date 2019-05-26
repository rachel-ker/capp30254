#!/bin/bash
#SBATCH --job-name=small_all_features
#SBATCH --output=example_sbatch_smallallfeatures.out
#SBATCH --error=example_sbatch_smallallfeatures.err
#SBATCH --partition=broadwl
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=14
#SBATCH --mem-per-cpu=2000

module load midway2; module load python/3.7.0
python3 run.py
