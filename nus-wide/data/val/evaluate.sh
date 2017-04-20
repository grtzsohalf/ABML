#!/bin/bash

for i in $(seq 10 1 25); do
    for j in $(seq 0.8 0.04 1); do
        python evaluate_beam_sh.py val.candidate.captions81_lr0.0005_bpmll-${i}_${j}.pkl \
        result_lr0.0005_bpmll-${i}_${j}.txt
    done
done
