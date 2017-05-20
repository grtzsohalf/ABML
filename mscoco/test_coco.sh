#!/bin/bash

for i in $(seq 10 10 100); do
    for j in $(seq 0.1 0.1 0.1); do
        python test_coco.py mscoco_init_pred-${i} mscoco_init_pred-${i} ${j}
    done
done
