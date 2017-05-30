#!/bin/bash

for i in $(seq 47 3 47); do
    for j in $(seq 0.35 0.1 0.35); do
        python test_coco.py mscoco_init_pred_concat-${i} mscoco_init_pred_concat-${i} ${j}
    done
done
