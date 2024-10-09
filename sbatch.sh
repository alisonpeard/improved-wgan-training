#!/bin/bash
#SBATCH --job-name=WGAN-GP
#SBATCH --output=sbatch/%A_%a.out
#SBATCH --error=sbatch/%A_%a.err
#SBATCH --partition=GPU
#SBATCH --gres=gpu:tesla:1
#SBATCH --time=24:00:00

python gan_mnist.py

# To run from shell:
# ------------------
# $ micromamba activate hazGAN
# $ sbatch sbatch.sh

# if not working with sbatch, run
# srun -p GPU --gres=gpu:tesla:1 --time=12:00:00 --pty python gan_mnist.py