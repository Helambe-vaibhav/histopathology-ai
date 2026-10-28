#!/usr/bin/env bash
# ==============================================================================
# Automated Ablation Ladder Queue Monitor
# Watches target process (PID 2810420), waits for termination & GPU VRAM release,
# and automatically starts the 5-stage ablation ladder.
# ==============================================================================
set -u

TARGET_PID=2810420
PROJECT_DIR="/home/vhelambe/tmp/histopathology-ai"
PYTHON_BIN="/home/vhelambe/tmp/gpu_env/bin/python"
LOG_FILE="$PROJECT_DIR/ablation_monitor.log"
ABLATION_LOG="$PROJECT_DIR/ablation.log"

cd "$PROJECT_DIR" || exit 1

ts() {
    date "+%Y-%m-%d %H:%M:%S"
}

echo "[$(ts)] ==========================================" > "$LOG_FILE"
echo "[$(ts)] Automated Queue Monitor Active." >> "$LOG_FILE"
echo "[$(ts)] Monitoring other user process PID: $TARGET_PID (train_modified.py)" >> "$LOG_FILE"
echo "[$(ts)] Will automatically trigger Ablation Ladder once PID $TARGET_PID terminates and GPU memory frees." >> "$LOG_FILE"
echo "[$(ts)] ==========================================" >> "$LOG_FILE"

# 1. Wait for process to exit
while ps -p "$TARGET_PID" >/dev/null 2>&1; do
    MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "N/A")
    echo "[$(ts)] PID $TARGET_PID is running (GPU VRAM in use: ${MEM_USED} MiB). Waiting 20s..." >> "$LOG_FILE"
    sleep 20
done

echo "[$(ts)] PID $TARGET_PID has terminated!" >> "$LOG_FILE"
echo "[$(ts)] Checking GPU memory reclamation..." >> "$LOG_FILE"

# 2. Wait until GPU memory drops below 1000 MiB
while true; do
    MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "0")
    if [ "$MEM_USED" -lt 1000 ]; then
        echo "[$(ts)] GPU memory cleared: ${MEM_USED} MiB in use. GPU 0 is fully available." >> "$LOG_FILE"
        break
    else
        echo "[$(ts)] GPU memory still recovering (${MEM_USED} MiB). Waiting 5s..." >> "$LOG_FILE"
        sleep 5
    fi
done

# 3. Extra safety buffer
sleep 5

# 4. Record baseline GPU info
nvidia-smi >> "$LOG_FILE"

# 5. Launch Ablation Ladder in offline mode
export HF_HUB_OFFLINE=1
echo "[$(ts)] LAUNCHING 5-STAGE ABLATION LADDER (M5 -> M4 -> M3 -> M2 -> M1)..." >> "$LOG_FILE"
$PYTHON_BIN scripts/run_ablation_ladder.py --config configs/config.yaml --stages M5 M4 M3 M2 M1 --epochs 35 > "$ABLATION_LOG" 2>&1
STATUS=$?

echo "[$(ts)] Ablation Ladder completed with exit status: $STATUS" >> "$LOG_FILE"
