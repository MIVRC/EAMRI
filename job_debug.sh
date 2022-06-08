#!/bin/bash

python model_train.py --exp-dir=./result/convTranNet_0601_debug_brain_ex3_batch8 --netType=convTranNet_0601_debug --dataName=cc359 --accer=4 --dataMode=complex_edge --batchSize=8

