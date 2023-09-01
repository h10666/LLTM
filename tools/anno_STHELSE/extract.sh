#!/bin/bash
#SBATCH -J practical
#SBATCH -p defq
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 7-00:00:00
#SBATCH --gres=gpu:0
#SBATCH -o outs/slurm-%j.out
module load cuda11.1/toolkit/11.1.1

python -u /home/10102007/TSM_NEW_CHECK3/tools/anno_STHELSE/gen_label_STHELSE.py
# python -u /home/10102007//TSM_NEW_CHECK3/tools/anno_STHELSE/shuffle.py
