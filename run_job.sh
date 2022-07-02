#!/bin/bash

#SBATCH -J 0702
#SBATCH -N 1 -c 8
#SBATCH --gres=gpu:2

python model_train.py --exp-dir=./result/MRI_egde/cc359_4x/net_0702_var2 --netType=net_0702_var2 --dataName=cc359 --accer=4 --dataMode=complex_edge --batchSize=8

