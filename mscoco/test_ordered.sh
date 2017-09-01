#!/bin/bash

for i in $(seq 10 2 20); do
    for j in $(seq 0.1 0.1 0.1); do
        python test_ordered.py mscoco_ordered-${i} mscoco_ordered-${i} ${j}
    done
done
