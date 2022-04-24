#!/bin/bash
python finetune.py --maxdisp 192 --model stackhourglass \
    --datapath /PSMNet/training/ \
    --loadmodel /PSMNet/pretrained_sceneflow_new.tar \
    --savemodel finesave --epochs 10
