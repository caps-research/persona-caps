#!/usr/bin/env bash
# Run π_d sweep for all 5 Big-Five dims on Qwen3.5-9B via remote Ollama (3090).
# Single-GPU = workers=1 (Ollama serializes by default).
set -u  # do NOT set -e so one failed dim doesn't block the others
cd "$(dirname "$0")/.."

export OLLAMA_HOST="${OLLAMA_HOST:-http://192.168.50.24:11434}"
export PYTHONPATH="src:${PYTHONPATH:-}"

LOG_DIR="runs/_logs"
mkdir -p "$LOG_DIR"

DIMS=(neuroticism extraversion conscientiousness agreeableness openness)
SHORT=(n e c a o)

for i in "${!DIMS[@]}"; do
    dim="${DIMS[$i]}"
    short="${SHORT[$i]}"
    out="runs/pi_sweep_${short}_q9b"
    log="$LOG_DIR/pi_sweep_${short}_q9b.log"

    if [[ -f "$out/summary.json" ]]; then
        echo "[skip] $dim already has $out/summary.json"
        continue
    fi

    echo "=== [$(date +%H:%M:%S)] starting $dim → $out ===" | tee -a "$log"
    PYTHONUNBUFFERED=1 python3 -u experiments/exp_pi_sweep.py \
        --target-dim "$dim" \
        --subject-backend ollama \
        --subject-model qwen3.5:latest \
        --subject-temp 0.7 \
        --top-k-per-dim 4 \
        --n-runs 3 \
        --workers 1 \
        --out-dir "$out" \
        2>&1 | tee -a "$log"
    echo "=== [$(date +%H:%M:%S)] finished $dim (rc=${PIPESTATUS[0]}) ===" | tee -a "$log"
done

echo
echo "All sweeps done. Summaries:"
for short in "${SHORT[@]}"; do
    f="runs/pi_sweep_${short}_q9b/summary.json"
    if [[ -f "$f" ]]; then
        echo "  $f"
    else
        echo "  $f  [MISSING]"
    fi
done
