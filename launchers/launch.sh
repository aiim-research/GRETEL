#!/bin/bash
#$ -S /bin/bash
#$ -pe mpi 24
#$ -cwd
#$ -o ./output/qsub/std_$JOB_ID.out
#$ -e ./output/qsub/err_$JOB_ID.out
#$ -q parallel.q


export OMP_NUM_THREADS=$NSLOTS
export OPENBLAS_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS
export VECLIB_MAXIMUM_THREADS=$NSLOTS
export NUMEXPR_NUM_THREADS=$NSLOTS

python $1 $2 $3
