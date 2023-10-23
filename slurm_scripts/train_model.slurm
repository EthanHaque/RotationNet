#!/bin/bash
#SBATCH --job=train_model
#SBATCH --output=logs/train_model%j.out
#SBATCH --error=logs/train_model%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=32G
#SBATCH --time=01:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eh0560@princeton.edu
#SBATCH --gres=gpu:1

module purge
module load anaconda3/2023.3
conda activate torch_env

python /scratch/gpfs/eh0560/SkewNet/src/SkewNet/model/train.py