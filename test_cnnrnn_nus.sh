#!/bin/bash

for i in $(seq 1 1 20); do
    for j in $(seq 0.0 0.1 0.0); do
        python test.py nus_cnnrnn-${i} nus_cnnrnn-${i} ${j}
    done
done
