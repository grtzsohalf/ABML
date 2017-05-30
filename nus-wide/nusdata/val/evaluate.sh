#!/bin/bash

for i in $(seq 20 3 20); do
    for j in $(seq 0.3 0.1 0.3); do
        python evaluate.py val.candidate.captions81_nus_init_pred-${i}_${j}.pkl \
        result_nus_init_pred-${i}_${j}.txt
    done
done
