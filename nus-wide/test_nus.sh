#!/bin/bash

for i in $(seq 20 3 20); do
    for j in $(seq 0.3 0.1 0.3); do
        python test_nus.py nus_init_pred-${i} nus_init_pred-${i} ${j}
    done
done
