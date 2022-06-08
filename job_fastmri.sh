#!/bin/bash

python model_train.py --exp-dir=./result/convTranNet_0601_fastmri --netType=convTranNet_0601_fastmri --dataName=fastmri --accer=4 --dataMode=complex_edge --batchSize=16 --resume=1

