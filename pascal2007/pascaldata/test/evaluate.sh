#!/bin/bash

for i in $(seq 10 10 90); do
    for j in $(seq 0.09 0.01 0.09); do
        python evaluate.py test.candidate.captions_pascal_0.001-${i}_${j}.pkl \
        result_pascal_0.001-${i}_${j}.txt
    done
done
