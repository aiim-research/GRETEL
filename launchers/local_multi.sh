#!/bin/bash
# Executes several (sequential) runs of the same configuration

cfg=config/TCR-500-64-0.4_GCN_RSGG.jsonc
num_runs=1

for i in {1..$num_runs}
do
    python main.py $cfg $i
done
