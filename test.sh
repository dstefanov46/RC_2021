#!/bin/bash

datasets=(traffic tourism tourismlarge labour)
methods=(HierE2E)

for dataset in "${datasets[@]}"; do
     for method in "${methods[@]}"; do
          python3 experiments/run_experiment_with_best_hps.py --dataset "$dataset" --method "$method" --num-runs 2
     done
done
