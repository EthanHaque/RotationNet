#!/bin/bash
#SBATCH --job=train_model
#SBATCH --output=logs/train_model%j.out
#SBATCH --error=logs/train_model%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eh0560@princeton.edu
#SBATCH --gres=gpu:2

module purge
module load anaconda3/2023.9
conda activate torch_env

export OMP_NUM_THREADS=2

torchrun \
--nnodes 1 \
--nproc_per_node 2 \
/scratch/gpfs/eh0560/SkewNet/src/SkewNet/model/train.py \
/scratch/gpfs/eh0560/SkewNet/model_training_configs/MobileNetV3Backbone.json