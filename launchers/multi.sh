#!/bin/bash
# Schedule multiple runs of all the configurations contained in the searchdir

search_dir=./config/_papers/ijcai24/main
num_runs=33

for i in {1..$num_runs}
do
    for entry in "$search_dir"/*.jsonc
    do
        qsub launchers/launch.sh main.py $entry $i
    done
done
