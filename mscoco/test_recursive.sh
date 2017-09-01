#!/bin/bash

for i in $(seq 75 5 85); do
    for j in $(seq 0.0 0.1 0.0); do
        python test_recursive.py mscoco_recursive-${i} mscoco_recursive-${i} ${j}
    done
done
