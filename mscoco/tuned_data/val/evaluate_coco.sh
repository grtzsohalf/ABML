#!/bin/bash

for i in $(seq 30 5 35); do
    python evaluate_coco.py val.candidate_mscoco-${i}_0.1.pkl \
    result_mscoco-${i}.txt
done
