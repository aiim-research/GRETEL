#!/bin/bash -l
#SBATCH -o output/logs/%j.out
#SBATCH -s
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J RSGG-MAIN
#SBATCH -c 8
#SBATCH -p normal

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun python "$@"
