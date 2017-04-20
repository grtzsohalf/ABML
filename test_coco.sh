#!/bin/bash

for i in $(seq 7 1 7); do
    for j in $(seq 0.2 0.1 0.2); do
        python test_coco.py mscoco_dropout0.8-${i} mscoco_dropout0.8-${i} ${j}
    done
done
