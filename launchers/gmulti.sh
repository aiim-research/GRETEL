#!/bin/bash

#search_dir=./config/optimus/clear-gpu-tc28
search_dir=./config/optimus/clear-bbbp

declare -a MIGS=(MIG-28012936-dae0-5a25-9417-32aec38a908b MIG-cbc43902-8045-5008-b0be-7a9289ec0c5e MIG-c1fe567e-0f72-59a0-87a1-d97939d2c8f4 MIG-f3f384f0-9f68-516e-b535-f8e0dcdeae41 MIG-b12fa2d9-c96e-5d0b-b8a2-fe9118e32619 MIG-35282ee9-3aa6-571d-9b9e-e8aba8d95121 MIG-7082f15e-b12d-504c-96e0-3b8a2aadd641)

curr=0
for i in {1..1}
do
    for entry in "$search_dir"/*
    do
        qsub launchers/glaunch.sh ${MIGS[$(( $curr % ${#MIGS[@]} ))]} main.py $entry $i
        ((curr++))
    done
done
