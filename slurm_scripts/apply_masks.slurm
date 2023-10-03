#!/bin/bash
#SBATCH --job=mask
#SBATCH --output=logs/mask%j.out
#SBATCH --error=logs/mask%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eh0560@princeton.edu

module purge
module load anaconda3/2023.3
conda activate torch_env

python /scratch/gpfs/eh0560/SkewNet/src/SkewNet/apply_segmentation_masks.py
