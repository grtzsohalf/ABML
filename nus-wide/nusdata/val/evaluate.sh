#!/bin/bash

for i in $(seq 30 10 30); do
    for j in $(seq 0.6 0.1 0.6); do
        python evaluate.py val.candidate.captions81_nus_init_pred-${i}_${j}.pkl \
        result_init_pred-${i}_${j}.txt
    done
done
