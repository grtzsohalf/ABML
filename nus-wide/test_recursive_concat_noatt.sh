#!/bin/bash

for i in $(seq 10 1 10); do
    for j in $(seq 0.06 0.04 0.1); do
        python test_recursive_concat_noatt.py nus_recursive_concat_noatt-${i} nus_recursive_concat_noatt-${i} ${j}
    done
done
