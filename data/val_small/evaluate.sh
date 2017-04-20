#!/bin/bash

for i in $(seq 18 1 18); do
    for j in $(seq 0.4 0.1 0.4); do
        python evaluate_beam_sh.py val_small.candidate.captions81_nus_dropout0.8-${i}_${j}.pkl \
        result_nus_dropout0.8-${i}_${j}.txt
    done
done
