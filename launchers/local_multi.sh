#!/bin/bash
# Executes several (sequential) runs of the same configuration

#cfg=config/_papers/ijcai24/baselines/baselines_asd.jsonc
cfg=$1

for i in {1..10}
do
    python main.py $cfg $i
done
