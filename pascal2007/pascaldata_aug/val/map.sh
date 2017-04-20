#!/bin/bash

for i in $(seq 50 1 50); do
    for j in $(seq 0.0 1 0.0); do
        python map.py val.candidate.captions_pascal_aug_ratio-${i}_${j}.pkl \
        map_pascal_aug_ratio-${i}_${j}.txt
    done
done
