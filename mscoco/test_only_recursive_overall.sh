#!/bin/bash

for i in $(seq 20 3 50); do
    for j in $(seq 0.1 0.1 0.5); do
        python test_only_recursive_overall.py mscoco_only_recursive-${i} mscoco_only_recursive-${i} ${j}
    done
done
