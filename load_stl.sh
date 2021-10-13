#!/bin/bash
#SBATCH -J load_stl
#SBATCH -o log_slurm.o%j
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -q gpu
#SBATCH -t 12:00:00

module load slurm
module load cuda10.0

python simclr.py --config_env configs/env.yml --config_exp configs/pretext/simclr_stl10.yml