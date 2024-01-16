#!/bin/bash
# Schedule multiple runs of all the configurations contained in the searchdir

search_dir=./config/_papers/ijcai24/baselines
MINWAIT=1
MAXWAIT=10


for i in {1..1}
do
    for entry in "$search_dir"/*.jsonc
    do
       sleep $((MINWAIT+RANDOM % (MAXWAIT-MINWAIT)))
       qsub launchers/launch.sh main.py $entry $i
    done
done
