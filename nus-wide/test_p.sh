#!/bin/bash

for i in $(seq 9 1 12); do
    for j in $(seq 0.1 0.1 0.1); do
        python test_p.py nus_p-${i} nus_p-${i} ${j}
    done
done
