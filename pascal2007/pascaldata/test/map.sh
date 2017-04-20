#!/bin/bash

for i in $(seq 290 50 290); do
    for j in $(seq 0.0 1 0.0); do
        python map_val.py test.candidate.captions_pascal_dropout-${i}_${j}.pkl \
        map_pascal_dropout-${i}_${j}.txt
    done
done
