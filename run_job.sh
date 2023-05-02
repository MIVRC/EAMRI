#!/bin/bash
# example of job script

CUDA_VISIBLE_DEVICES=0 python model_train.py \
                                        --seed=43 \
                                        --netType=EAMRI \
                                        --batchSize=4 \
                                        --num_epochs=80 \
                                        --accer=4 \
                                        --center_fractions=0.08 \
                                        --dataName=cc359 \
                                        --challenge=multicoil \
                                        --dataMode=complex
                                        --exp_dir=./result/EAMRI/ \
                                        --train_root="put your own train root here" \
                                        --valid_root="put your own train root here" \
                                        --use_sens_map=0 \
