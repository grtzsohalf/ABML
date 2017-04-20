#!/bin/bash

for i in $(seq 450 1 450); do
    for j in $(seq 0.0 1 0.0); do
        python map.py val.candidate.captions_pascal_0.0005-${i}_${j}.pkl \
        map_pascal_0.0005-${i}_${j}.txt
    done
done
