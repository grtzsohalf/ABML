#!/bin/bash

for i in $(seq 5 5 5); do
    for j in $(seq 0.1 0.1 0.1); do
        python test_recursive_concat.py mscoco_recursive_concat-${i} mscoco_recursive_concat-${i} ${j}
    done
done
