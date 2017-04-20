#!/bin/bash

for i in $(seq 40 1 40); do
    for j in $(seq 0.0 0.1 0.0); do
        python test_pascal.py pascal_emb-${i} pascal_emb-${i} ${j}
    done
done
