#!/bin/bash

for i in $(seq 300 1 300); do
    for j in $(seq 0.0 0.1 0.0); do
        python test_pascal.py pascal_init_pred-${i} pascal_init_pred-${i} ${j}
    done
done
