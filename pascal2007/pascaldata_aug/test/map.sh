#!/bin/bash

for i in $(seq 100 1 100); do
    for j in $(seq 0.0 1 0.0); do
        python map.py test.candidate.captions_pascal_aug_0.0005-${i}_${j}.pkl \
        map_pascal_aug_0.0005-${i}_${j}.txt
    done
done
