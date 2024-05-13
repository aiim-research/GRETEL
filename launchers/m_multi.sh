#!/bin/bash
# Schedule multiple runs of all the configurations contained in the searchdir

search_dir=./config/ensembles/TCR_512_32_0.25-TCO-BaselineEnsembles/25-TCO-Ens[OBS+2xiRand+2xRSGG]-Baselines_do0_e4
MINWAIT=1
MAXWAIT=10


for i in {1..1}
do
    for entry in "$search_dir"/*.json
    do
       sleep $((MINWAIT+RANDOM % (MAXWAIT-MINWAIT)))
       sbatch launchers/m_launch.sh main.py $entry $i
    done
done