#!/bin/bash

for i in $(seq 4 1 20); do
    for j in $(seq 0.0 0.1 0.0); do
        python evaluate.py val.candidate.captions81_nus_cnnrnn-${i}.pkl \
        result_nus_cnnrnn-${i}.txt
    done
done
