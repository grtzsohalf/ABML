#!/bin/bash

for i in $(seq 18 1 18); do
    for j in $(seq 0.4 0.1 0.5); do
        python test.py nus_dropout0.8-${i} nus_dropout0.8-${i} ${j}
    done
done
