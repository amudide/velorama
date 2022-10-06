#!/bin/bash

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMOP_NUM_THREADS=1

export CUDA_VISIBLE_DEVICES=0,1,2,3

# use this to name output files (automatically already includes num. of hidden dimensions, dynamics, lam, lam_ridge, lag in output file name)
method="baseline.lr1e-4" 

hidden=16
lr=0.0001
L=5
max_iter=5000
p="H"
trial_no=0
tol=0.01
device="cuda"

for dataset in "De-noised_100G_6T_300cPerT_dynamics_7_DS6" "De-noised_100G_3T_300cPerT_dynamics_9_DS4" # "De-noised_100G_4T_300cPerT_dynamics_10_DS5" "De-noised_100G_7T_300cPerT_dynamics_11_DS7"
    do
    for num in 0 1 2 # 3 4 # 5 6 7 8 9 10 11 12 13 14
        do
            dataset_name="${dataset}_${num}"
            echo $dataset_name
            python -u /data/cb/alexwu/tf_lag/run.py -m $method -ds $dataset_name -dyn "pseudotime" -dev $device -tn $trial_no -lmr 0 -p $p -l $L -hd $hidden -mi $max_iter -lr $lr -tol $tol
            python -u /data/cb/alexwu/tf_lag/run.py -m $method -ds $dataset_name -dyn "rna_velocity" -dev $device -tn $trial_no -lmr 0 -p $p -l $L -hd $hidden -mi $max_iter -lr $lr -tol $tol
        done
    done
