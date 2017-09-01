#!/bin/bash

for i in $(seq 10 2 20); do
    for j in $(seq 0.1 0.1 0.1); do
        python evaluate.py val.candidate.captions_mscoco_ordered-${i}_${j}.pkl \
        result_ordered-${i}_${j}.txt
    done
done
