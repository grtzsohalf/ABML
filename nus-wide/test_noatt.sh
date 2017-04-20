#!/bin/bash

for i in $(seq 8 1 8); do
    for j in $(seq 0.1 0.1 0.9); do
        python test_noatt.py nus_noatt-${i} nus_noatt-${i} ${j}
    done
done
