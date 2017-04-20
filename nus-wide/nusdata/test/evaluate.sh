#!/bin/bash

for i in $(seq 8 1 8); do
    for j in $(seq 0.1 0.1 0.9); do
        python evaluate.py test.candidate.captions81_nus_noatt-${i}_${j}.pkl \
        result_nus_noatt-${i}_${j}.txt
    done
done
