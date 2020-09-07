#!/bin/bash

#SBATCH -o ./segment_hq_2d_out.txt
#SBATCH -e ./segment_hq_2d_error.txt
#SBATCH -J segment_2d_hq
#SBATCH --mem=4G
#SBATCH -p icb_gpu
#SBATCH --gres=gpu:1
#SBATCH -t 40:00:00
#SBATCH --nice=100

source activate tf2
python train.py
