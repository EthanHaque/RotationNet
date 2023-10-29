#!/bin/bash
#SBATCH --job=train_model
#SBATCH --output=logs/train_model%j.out
#SBATCH --error=logs/train_model%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=168G
#SBATCH --time=05:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eh0560@princeton.edu
#SBATCH --gres=gpu:1

module purge
module load anaconda3/2023.3
conda activate torch_env

python /scratch/gpfs/eh0560/SkewNet/src/SkewNet/model/train.py