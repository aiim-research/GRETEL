#!/bin/bash
#$ -S /bin/bash
#$ -pe mpi 1
#$ -cwd
#$ -o ./output/qsub/std_$JOB_ID.out
#$ -e ./output/qsub/err_$JOB_ID.out
#$ -q gpu4.q
mkdir -p ./output/qsub

export CUDA_VISIBLE_DEVICES=$1
echo $CUDA_VISIBLE_DEVICES

python $2 $3 $4
