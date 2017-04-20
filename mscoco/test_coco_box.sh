#!/bin/bash

for i in $(seq 20 1 20); do
    for j in $(seq 0.3 0.1 0.3); do
        python test_coco_box.py mscoco_lr0.001-${i} mscoco_lr0.001-${i} ${j}
    done
done
