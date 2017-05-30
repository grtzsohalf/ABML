#!/bin/bash

for i in $(seq 50 10 50); do
    for j in $(seq 0.0 0.1 0.0); do
        python test_recursive_feature.py mscoco_recursive_feature-${i} mscoco_recursive_feature-${i} ${j}
    done
done
