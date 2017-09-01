#!/bin/bash

for i in $(seq 8 1 8); do
    for j in $(seq 0.1 0.1 0.1); do
        python test_recursive_concat_orderfreebeam.py nus_recursive_concat-${i} nus_recursive_concat_orderfreebeam-${i} ${j}
    done
done
