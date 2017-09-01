#!/bin/bash

for i in $(seq 150 20 270); do
    for j in $(seq 0.3 0.1 0.3); do
        python test_pascal.py pascal_y-${i} pascal_y-${i} ${j}
    done
done
