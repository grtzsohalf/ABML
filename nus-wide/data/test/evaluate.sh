#!/bin/bash

for i in $(seq 23 1 23); do
    for j in $(seq 0.4 0.1 0.5); do
        python evaluate.py test.candidate.captions81_nus_dropout0.8-${i}_${j}.pkl \
        result_nus_dropout0.8-${i}_${j}.txt
    done
done
