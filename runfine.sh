#!/bin/bash
python finetune.py --maxdisp 192 --model stackhourglass \
    --datapath /PSMNet/training/ \
    --loadmodel /PSMNet/pretrained_model_KITTI2015.tar \
    --savemodel finesave --epochs 1
