#!/bin/bash

for i in $(seq 25 1 25); do
    for j in $(seq 0.2 0.1 0.2); do
        python test_noatt.py nus_noatt-${i} nus_noatt-${i} ${j}
    done
done
