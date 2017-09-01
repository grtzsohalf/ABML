#!/bin/bash

for i in $(seq 15 5 15); do
    for j in $(seq 0.18 0.01 0.19); do
        python test_p.py mscoco_p-${i} mscoco_p-${i} ${j}
    done
done
