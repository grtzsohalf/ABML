#!/bin/bash

for i in $(seq 7 1 7); do
    for j in $(seq 0.1 0.1 0.3); do
        python evaluate_coco.py val_small.candidate.captions_mscoco_dropout0.8-${i}_${j}.pkl \
        result_mscoco_dropout0.8-${i}_${j}.txt
    done
done
