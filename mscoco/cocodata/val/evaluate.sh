#!/bin/bash

for i in $(seq 47 3 47); do
    for j in $(seq 0.35 0.1 0.35); do
        python evaluate.py val.candidate.captions_mscoco_init_pred_concat-${i}_${j}.pkl \
        result_init_pred_concat-${i}_${j}.txt
    done
done
