#!/bin/bash

for i in $(seq 5 5 200); do
    for j in $(seq 0.0 0.1 0.0); do
        python test_cnnrnn_pascal.py pascal_cnnrnn-${i} pascal_cnnrnn-${i} ${j}
    done
done
