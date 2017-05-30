#!/bin/bash

for i in $(seq 250 1 250); do
    for j in $(seq 0.0 0.1 0.0); do
        python test_pascal.py iterative_update-${i} iterative_update-${i} ${j}
    done
done
