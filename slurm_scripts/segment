#!/bin/bash
#SBATCH --job=segment
#SBATCH --output=logs/segment%j.out
#SBATCH --error=logs/segment%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eh0560@princeton.edu
#SBATCH --gres=gpu:1

module purge
module load anaconda3/2023.3
conda activate torch_env

python /scratch/gpfs/eh0560/SkewNet/src/SkewNet/create_segmentation_masks.py
