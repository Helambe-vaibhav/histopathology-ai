#!/usr/bin/env bash
cd /home/vhelambe/tmp/histopathology-ai

PID=$(cat train_100ep.pid)
echo "[WATCHER] Monitoring training PID $PID..."

while kill -0 "$PID" 2>/dev/null; do
    sleep 30
done

echo "[WATCHER] Training complete! Launching calibrated evaluation on test cohort..."
/home/vhelambe/tmp/gpu_env/bin/python src/evaluate.py \
    --config configs/config.yaml \
    --weights results/checkpoints/unet_production_100ep.pth \
    --output-csv eval_production_100ep.csv \
    --threshold 0.15 \
    --min-distance 8 > eval_production_100ep.log 2>&1

echo "[WATCHER] Calibrated evaluation complete! Scorecard saved to eval_production_100ep.csv"
