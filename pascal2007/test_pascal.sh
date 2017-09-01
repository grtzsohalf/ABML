#!/bin/bash

for i in $(seq 250 10 280); do
    for j in $(seq 0.3 0.1 0.3); do
        python test_pascal.py iterative_update-${i} iterative_update-${i} ${j}
    done
done
