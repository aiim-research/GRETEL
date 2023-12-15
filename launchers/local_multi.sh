#!/bin/bash
# Executes several (sequential) runs of the same configuration

cfg=config/TCR-500-28-0.3_GCN_RSGG.jsonc
num_runs=33

for i in {1..$num_runs}
do
    python main.py $cfg $i
done
