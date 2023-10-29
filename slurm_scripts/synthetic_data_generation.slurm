#!/bin/bash
#SBATCH --job=synthetic_data_generation
#SBATCH --output=logs/synthetic_data_generation%j.out
#SBATCH --error=logs/synthetic_data_generation%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=32G
#SBATCH --time=11:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eh0560@princeton.edu

module purge
module load anaconda3/2023.3
conda activate torch_env

python /scratch/gpfs/eh0560/SkewNet/src/SkewNet/synthetic_data_generation.py
