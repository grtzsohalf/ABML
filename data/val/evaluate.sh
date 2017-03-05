#!/bin/bash

for i in $(seq 1 1 3); do
    for j in $(seq 0.1 0.1 0.9); do
        python evaluate_beam_sh.py val.candidate.captions81_lr0.0005_bpmll-${i}_${j}.pkl \
        result_lr0.0005_bpmll-${i}_${j}.txt
    done
done
