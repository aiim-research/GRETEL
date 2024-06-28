#!/bin/bash
#SBATCH -o /NFSHOME/lgutierrez/projects/GRETEL/output/multi-criteria-logs.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -J multi-criteria-test
#SBATCH -p normal

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun python "$@"
