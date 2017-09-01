#!/bin/bash

for i in $(seq 9 1 9); do
    for j in $(seq 0.25 0.01 0.25); do
        python evaluate.py test.candidate.captions_mscoco_recursive_concat-${i}_${j}.pkl \
        result_mscoco_recursive_concat-${i}_${j}.txt
    done
done
