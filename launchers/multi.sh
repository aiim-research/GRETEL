#!/bin/bash
# Schedule multiple runs of all the configurations contained in the searchdir

search_dir=./config/explainers/cf2
num_runs=1

for i in {1..$num_runs}
do
    for entry in "$search_dir"/*
    do
        qsub launchers/launch.sh main.py $entry $i
    done
done
