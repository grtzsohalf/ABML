#!/bin/bash

for i in $(seq 420 20 500); do
    for j in $(seq 0.3 0.1 0.3); do
        python test_recursive_concat.py pascal_recursive_concat-${i} pascal_recursive_concat-${i} ${j}
    done
done
