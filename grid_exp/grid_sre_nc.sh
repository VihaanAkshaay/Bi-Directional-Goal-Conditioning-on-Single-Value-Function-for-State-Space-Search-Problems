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
#SBATCH --output=sre_nc.txt
#SBATCH --error=sre_nc.txt

conda activate earthai
CUDA_VISIBLE_DEVICES=1 python exp_grid_sre_nc.py