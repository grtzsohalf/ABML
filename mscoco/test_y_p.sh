#!/bin/bash

for i in $(seq 10 1 10); do
    for j in $(seq 0.2 0.1 0.2); do
        python test_y_p.py mscoco_y_p-${i} mscoco_y_p-${i} ${j}
    done
done
