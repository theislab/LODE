#!/bin/bash

#SBATCH -o ./segment1_out.txt
#SBATCH -e ./segment1_error.txt
#SBATCH -J segment1
#SBATCH --mem=2G
#SBATCH -p icb_gpu
#SBATCH --gres=gpu:1
#SBATCH -t 40:00:00
#SBATCH --nice=100

source activate tf_1.14
python segment.py "model_v1" "test"