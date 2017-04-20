#!/bin/bash

for i in $(seq 7 1 7); do
    for j in $(seq 0.2 0.1 0.2); do
        python evaluate_coco.py test.candidate.captions_mscoco_aug-${i}_${j}.pkl \
        result_mscoco_aug-${i}_${j}.txt
    done
done
