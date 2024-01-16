#!/bin/bash
# Delete Multiple contiguos scheduled job

jST=127356
jEND=127408


for i in {127356..127408}
do
    qdel $i
done
