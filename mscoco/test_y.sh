#!/bin/bash

for i in $(seq 5 5 25); do
    for j in $(seq 0.3 0.1 0.3); do
        python test_y.py mscoco_y-${i} mscoco_y-${i} ${j}
    done
done
