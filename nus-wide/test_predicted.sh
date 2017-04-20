#!/bin/bash

for i in $(seq 4 1 4); do
    for j in $(seq 0.1 0.1 0.3); do
        python test_pascal.py mscoco_pascal_0.0001-${i} mscoco_pascal_0.0001-${i} ${j}
    done
done
