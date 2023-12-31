#!/usr/bin/env bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --gpus=1
#SBATCH --partition=debug
#SBATCH --time=2-2:34:56 
#SBATCH --account=vihaan
#SBATCH --mail-type=all 
#SBATCH --mail-user=vihaanakshaay@ucsb.edu 
#SBATCH --output=re.txt
#SBATCH --error=re.txt

conda activate earthai
CUDA_VISIBLE_DEVICES=2 python exp_toh_re.py