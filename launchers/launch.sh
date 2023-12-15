#!/bin/bash
#$ -S /bin/bash
#$ -pe mpi 4
#$ -cwd
#$ -o ./output/qsub/std_$JOB_ID.out
#$ -e ./output/qsub/err_$JOB_ID.out
#$ -q parallel.q
mkdir -p ./output/qsub

python $1 $2 $3
