#!/bin/bash

for i in $(seq 300 50 500); do
    for j in $(seq 0.09 0.1 0.09); do
        python evaluate.py val.candidate.captions_pascal_0.002-${i}_${j}.pkl \
        result_pascal_0.002-${i}_${j}.txt
    done
done
