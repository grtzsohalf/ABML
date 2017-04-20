#!/bin/bash

for i in $(seq 1 1 20); do
    for j in $(seq 0.0 0.1 0.0); do
        python evaluate_coco.py val.candidate.captions_coco_cnnrnn-${i}.pkl \
        result_coco_cnnrnn-${i}.txt
    done
done
