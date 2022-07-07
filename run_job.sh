#!/bin/bash

#SBATCH -J var4
#SBATCH -N 1 -c 11
#SBATCH --gres=gpu:2

python model_train.py --exp-dir=./result/MRI_egde/cc359_4x/net_0707_var2 --netType=net_0707_var2 --dataName=cc359 --accer=4 --dataMode=complex_edge --batchSize=16 --server=ai --is_evaluate=0

