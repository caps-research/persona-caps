#!/usr/bin/env bash
# Extension grid for the §4.7 GPT-4o cross-vendor pilot.
# Runs 5 additional conditions on top of the initial high_E / K=15 pilot:
#   * high_E  K∈{5,10}     -- K-saturation stress test
#   * low_A  K=15          -- 2nd cell
#   * high_N K=15          -- 3rd cell
#   * low_O  K=15          -- 4th cell  (CAPS-only; P²-sanity will fail)
# Total wall-clock estimate: ~45 min serial.

set -uo pipefail

cd "$(dirname "$0")/../.."

LOG=runs/audit_loop_xvendor/_extend.log
echo "[extend] start at $(date -u +%FT%TZ)" | tee -a "$LOG"

run_one () {
  local cell="$1" dim="$2" pol="$3" k="$4"
  echo "[extend] === $cell K=$k ===" | tee -a "$LOG"
  python scripts/audit_loop/run_xvendor_pilot.py \
    --cell "$cell" --targeted-dim "$dim" --polarity "$pol" \
    --k "$k" --seeds 1 2 3 \
    --subject openai:gpt-4o-2024-08-06 \
    --workers 6 \
    2>&1 | tee -a "$LOG"
}

run_one high_extraversion   extraversion       high  5
run_one high_extraversion   extraversion       high  10
run_one low_agreeableness   agreeableness      low   15
run_one high_neuroticism    neuroticism        high  15
run_one low_openness        openness           low   15

echo "[extend] done at $(date -u +%FT%TZ)" | tee -a "$LOG"
