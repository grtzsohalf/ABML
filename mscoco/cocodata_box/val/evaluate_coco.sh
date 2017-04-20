#!/bin/bash

for i in $(seq 20 1 20); do
    for j in $(seq 0.3 0.1 0.3); do
        python evaluate_coco.py val.candidate.captions_mscoco_lr0.001-${i}_${j}.pkl \
        result_mscoco_lr0.001-${i}_${j}.txt
    done
done
