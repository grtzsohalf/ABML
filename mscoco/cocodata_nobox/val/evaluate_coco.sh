#!/bin/bash

for i in $(seq 5 1 5); do
    for j in $(seq 0.1 0.1 0.1); do
        python evaluate_coco.py val.candidate.captions_mscoco_aug-${i}_${j}.pkl \
        result_mscoco_aug-${i}_${j}.txt
    done
done
