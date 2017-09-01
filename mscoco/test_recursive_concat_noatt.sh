#!/bin/bash

for i in $(seq 30 1 30); do
    for j in $(seq 0.05 0.01 0.1); do
        python test_recursive_concat_noatt.py mscoco_recursive_concat_noatt-${i} mscoco_recursive_concat_noatt-${i} ${j}
    done
done
