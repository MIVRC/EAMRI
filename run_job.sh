#!/bin/bash

#SBATCH -J var4
#SBATCH -N 1 -c 11
#SBATCH --gres=gpu:2

python model_train.py --exp-dir=./result/MRI_egde/cc359_multicoil/cartesian_4x/dccnn --netType=DCCNN_cc359_multicoil --dataName=cc359 --accer=4 --dataMode=complex --batchSize=8 --server=ai --challenge=multicoil 

