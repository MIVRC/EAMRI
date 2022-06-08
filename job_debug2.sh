#!/bin/bash

python model_train.py --exp-dir=./result/convTranNet_0601_debug2_brain_ex2_batch8 --netType=convTranNet_0601_debug2 --dataName=cc359 --accer=4 --dataMode=complex --batchSize=8

