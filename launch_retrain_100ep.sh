#!/usr/bin/env bash
set -e

cd /home/vhelambe/tmp/histopathology-ai
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

echo "=========================================================="
echo " Starting 100-Epoch Production Training Run on GPU 0      "
echo " Loss: Combined (70% Dice, 30% BCE)                       "
echo " Augmentation: Spatial D4 + Histology Stain/Hue/Sat Jitter"
echo " Target Checkpoint: unet_production_100ep.pth             "
echo "=========================================================="

nohup /home/vhelambe/tmp/gpu_env/bin/python src/train.py \
    --config configs/config.yaml \
    --epochs 100 \
    --loss combined \
    --ckpt-name unet_production_100ep.pth \
    > training_100ep.log 2>&1 &

PID=$!
echo $PID > train_100ep.pid
echo "TRAINING_LAUNCHED PID: $PID"
echo "Tailing log file..."
sleep 2
cat training_100ep.log | head -n 30
