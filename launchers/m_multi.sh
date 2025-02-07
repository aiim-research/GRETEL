#!/bin/bash
# Schedule multiple runs of all the configurations contained in the searchdir

search_dir=lab/config/meta_ens/bbbp-gcn/base_explainers_t0
MINWAIT=1
MAXWAIT=10


for i in {1..1}
do
    for entry in "$search_dir"/*.json
    do
       sleep $((MINWAIT+RANDOM % (MAXWAIT-MINWAIT)))
       sbatch launchers/m_launch.sh future_main.py $entry $i
    done
done