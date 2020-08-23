#!/bin/bash

#sBATCH -o ./segment3_out.txt
#SBATCH -e ./segment3_error.txt
#SBATCH -J segment3
#SBATCH --mem=2G
#SBATCH -p icb_gpu
#SBATCH --gres=gpu:1
#SBATCH -t 40:00:00
#SBATCH --nice=100

source activate tf2
python segment.py "model_v1" "paths3"
~                                   
