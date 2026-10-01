#!/bin/bash
cd /home/ubuntu/fairness-auditor
source /home/ubuntu/fairness-auditor/venv/bin/activate

MODEL=SASRec
DATASET=ml-32m-50k

LATEST="/home/ubuntu/fairness-auditor/saved/${MODEL}-${DATASET}-latest.pth"

if [ -f "$LATEST" ]; then
    echo "Found checkpoint: $LATEST — resuming"
    python /home/ubuntu/fairness-auditor/train.py --model $MODEL --dataset $DATASET --resume "$LATEST"
else
    echo "No checkpoint found for $MODEL on $DATASET — starting fresh"
    python /home/ubuntu/fairness-auditor/train.py --model $MODEL --dataset $DATASET
fi
