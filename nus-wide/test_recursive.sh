#!/bin/bash

for i in $(seq 20 10 50); do
    for j in $(seq 0.3 0.1 0.3); do
        python test_recursive.py nus_recursive-${i} nus_recursive-${i} ${j}
    done
done
