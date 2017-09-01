#!/bin/bash

for i in $(seq 30 5 45); do
    for j in $(seq 0.3 0.1 0.3); do
        python test_only_recursive.py mscoco_only_recursive-${i} mscoco_only_recursive-${i} ${j}
    done
done
