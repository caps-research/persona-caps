#!/usr/bin/env bash
# Full Llama-3-8B evaluation pipeline:
#   1) π_d sweep (5 dims × 5 weights × 3 runs)            ≈ 30 min
#   2) PersonalityBench head-to-head: CAPS + SP, 5 traits × 2 polarities × 459 q
#      ≈ 8 hours
#   3) Judge stage handled separately by judge_personalitybench.py
#
# Single-GPU 3090 → workers=1; PYTHONUNBUFFERED so the master log streams live.
set -u  # do NOT set -e — one failed cell shouldn't abort the rest
cd "$(dirname "$0")/.."

export OLLAMA_HOST="${OLLAMA_HOST:-http://192.168.50.24:11434}"
export PYTHONPATH="src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

LOG_DIR="runs/_logs"
mkdir -p "$LOG_DIR"

DIMS=(neuroticism extraversion conscientiousness agreeableness openness)
SHORT=(n e c a o)

###############################################################################
# Stage 1 — π_d sweep
###############################################################################
echo "=========================================================================="
echo "[$(date '+%F %H:%M:%S')] STAGE 1: π_d sweep on llama3:8b"
echo "=========================================================================="
for i in "${!DIMS[@]}"; do
    dim="${DIMS[$i]}"
    short="${SHORT[$i]}"
    out="runs/pi_sweep_${short}_l8b"
    log="$LOG_DIR/pi_sweep_${short}_l8b.log"

    if [[ -f "$out/summary.json" ]]; then
        echo "[skip] $dim already has $out/summary.json"
        continue
    fi

    echo "--- [$(date '+%H:%M:%S')] starting $dim → $out ---"
    python3 -u experiments/exp_pi_sweep.py \
        --target-dim "$dim" \
        --subject-backend ollama \
        --subject-model llama3:8b \
        --subject-temp 0.7 \
        --top-k-per-dim 4 \
        --n-runs 3 \
        --workers 1 \
        --out-dir "$out" \
        2>&1 | tee "$log"
    echo "--- [$(date '+%H:%M:%S')] finished $dim ---"
done

###############################################################################
# Stage 2 — PersonalityBench head-to-head (CAPS + SP)
###############################################################################
echo
echo "=========================================================================="
echo "[$(date '+%F %H:%M:%S')] STAGE 2: PersonalityBench on llama3:8b"
echo "=========================================================================="
PB_OUT="runs/personalitybench_l8b"

# --limit 50 per (trait, pol) keeps wall-time ≤ ~10h on a single 3090.
# Standard IRiS-protocol per-trait sample size is 50–100; 50 is sufficient
# for ±1.0/10 95% CI on the per-trait mean (binomial CI on a 1–5 scale).
PB_PER_TRAIT_LIMIT=50

for method in caps sp; do
    log="$LOG_DIR/pb_l8b_${method}.log"
    echo "--- [$(date '+%H:%M:%S')] starting PB / $method (--limit $PB_PER_TRAIT_LIMIT) ---"
    python3 -u experiments/exp_personalitybench.py \
        --method "$method" \
        --backend ollama \
        --model llama3:8b \
        --workers 1 \
        --limit "$PB_PER_TRAIT_LIMIT" \
        --out-dir "$PB_OUT" \
        2>&1 | tee "$log"
    echo "--- [$(date '+%H:%M:%S')] finished PB / $method ---"
done

echo
echo "=========================================================================="
echo "[$(date '+%F %H:%M:%S')] ALL STAGES DONE"
echo "=========================================================================="
echo "Summaries:"
for short in "${SHORT[@]}"; do
    f="runs/pi_sweep_${short}_l8b/summary.json"
    [[ -f "$f" ]] && echo "  ✓ $f" || echo "  ✗ $f MISSING"
done
echo "PB outputs:"
ls -la "$PB_OUT"/*/  2>/dev/null | head -20
