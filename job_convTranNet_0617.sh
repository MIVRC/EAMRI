#!/bin/bash

python model_train.py --exp-dir=./result/MRI_egde/cc359_4x/convTranNet0617 --netType=convTranNet_0617 --dataName=cc359 --accer=4 --dataMode=complex_edge --batchSize=8

