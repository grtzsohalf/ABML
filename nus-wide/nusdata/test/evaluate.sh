#!/bin/bash

for i in $(seq 9 1 9); do
    for j in $(seq 0.1 0.1 0.1); do
        python evaluate.py test.candidate.captions81_nus_recursive_concat-${i}_${j}.pkl \
        result_nus_recursive_concat-${i}_${j}.txt
    done
done
