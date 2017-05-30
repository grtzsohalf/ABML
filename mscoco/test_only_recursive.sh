#!/bin/bash

for i in $(seq 30 10 50); do
    for j in $(seq 0.0 0.1 0.0); do
        python test_only_recursive.py mscoco_only_recursive-${i} mscoco_only_recursive-${i} ${j}
    done
done
