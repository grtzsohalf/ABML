#!/bin/bash

for i in $(seq 30 10 30); do
    for j in $(seq 0.6 0.1 0.6); do
        python test_nus.py nus_init_pred-${i} nus_init_pred-${i} ${j}
    done
done
