#!/bin/bash

for i in $(seq 50 1 50); do
    for j in $(seq 0.0 0.1 0.0); do
        python test_pascal_aug.py pascal_aug_ratio-${i} pascal_aug_ratio-${i} ${j}
    done
done
