#!/bin/bash

#SBATCH -J mri-edge
#SBATCH -N 1 -c 11
#SBATCH --gres=gpu:2

python model_train.py --exp-dir=./result/MRI_egde/cc359_4x/convTranNet0601 --netType=convTranNet_0601 --dataName=cc359 --accer=4 --dataMode=complex_edge --batchSize=8

