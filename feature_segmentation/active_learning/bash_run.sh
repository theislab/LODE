#!/bin/bash

#SBATCH -o ./out_al.txt
#SBATCH -e ./error_al.txt
#SBATCH -J ae
#SBATCH --mem=2G
#SBATCH -p icb_cpu
#SBATCH -t 40:00:00
#SBATCH --nice=100

source activate tf_1.14
python oct_selection.py

