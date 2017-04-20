#!/bin/bash

for i in $(seq 5 5 200); do
    for j in $(seq 0.0 1 0.0); do
        python map.py val.candidate.captions_pascal_cnnrnn-${i}.pkl \
        map_pascal_cnnrnn-${i}.txt
    done
done
