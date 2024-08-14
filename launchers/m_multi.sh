#!/bin/bash
# Schedule multiple runs of all the configurations contained in the searchdir

search_dir=./lab/config/ensembles/qualitative
MINWAIT=1
MAXWAIT=10


for i in {1..1}
do
    for entry in "$search_dir"/*.jsonc
    do
       sleep $((MINWAIT+RANDOM % (MAXWAIT-MINWAIT)))
       sbatch launchers/m_launch.sh future_main.py $entry $i
    done
done