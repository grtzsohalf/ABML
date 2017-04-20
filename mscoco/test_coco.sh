#!/bin/bash

for i in $(seq 50 1 50); do
    for j in $(seq 0.1 0.1 0.1); do
        python test_pascal.py mscoco-${i} mscoco-${i} ${j}
    done
done
