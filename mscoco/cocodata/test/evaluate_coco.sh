#!/bin/bash

for i in $(seq 40 5 40); do
    python evaluate_github.py test.candidate_mscoco-${i}_0.1.pkl \
    result_coco-${i}.txt
done
