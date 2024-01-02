#!/bin/bash
#$ -S /bin/bash
#$ -pe mpi 16
#$ -cwd
#$ -o ./output/qsub/std_$JOB_ID.out
#$ -e ./output/qsub/err_$JOB_ID.out
#$ -q gpu4.q
mkdir -p ./output/qsub

export CUDA_VISIBLE_DEVICES=$1
echo $CUDA_VISIBLE_DEVICES

export OMP_NUM_THREADS=$NSLOTS
export OPENBLAS_NUM_THREADS=$NSLOTS
export MKL_NUM_THREADS=$NSLOTS
export VECLIB_MAXIMUM_THREADS=$NSLOTS
export NUMEXPR_NUM_THREADS=$NSLOTS

python $2 $3 $4
