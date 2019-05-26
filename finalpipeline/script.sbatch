#!/bin/bash
#SBATCH --job-name=mlpipeline
#SBATCH --output=example_sbatch.out
#SBATCH --error=example_sbatch.err
#SBATCH --partition=broadwl
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=14
#SBATCH --mem-per-cpu=2000

module load midway2; module load python/3.7.0
python3 run.py