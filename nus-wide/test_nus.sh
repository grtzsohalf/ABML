#!/bin/bash

for i in $(seq 11 1 11); do
    for j in $(seq 0.4 0.1 0.4); do
        python test_nus.py nus_aug-${i} nus_aug-${i} ${j}
    done
done
