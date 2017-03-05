#!/bin/bash

python test_predicted.py lr0.0005_beam_predicted_norm-4 lr0.0005_beam_predicted_norm-4 0.9
for i in {5..32..1}; do
    for j in $(seq 0.1 0.1 0.9); do
        python test_predicted.py lr0.0005_beam_predicted_norm-${i} lr0.0005_beam_predicted_norm-${i} ${j}
    done
done
