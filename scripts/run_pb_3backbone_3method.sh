#!/usr/bin/env bash
# scripts/run_pb_3backbone_3method.sh
#
# Reproducibility anchor for §4.4 (PersonalityBench head-to-head):
# 3 subject backbones × 3 prompting methods × 1 judge = 9 runs.
#
# Subjects:
#   minimax  — MiniMax-M2.7-highspeed (closed API)
#   gemini   — Gemini-2.5-Flash       (closed API)
#   llama3   — llama3:8b              (Q4_K_M via Ollama on a single 3090)
#
# Methods:
#   caps     — Persona-CAPS retrieval over Beck-2022 atoms (this work)
#   sp       — IRiS Simple-Prompt baseline (verbatim from NPTI/baseline_prompt.py)
#   p2       — IRiS P² paragraph-baseline (verbatim 800-description bank)
#
# Judge (held constant):
#   gpt-4o-2024-08-06, T=0, IRiS facet prompt
#
# Question budget (held constant):
#   --limit 50  (questions per (trait, polarity) cell; 5 traits × 2 pols × 50 = 500 / cfg)
#   This makes all 9 cells of the 3×3 table apples-to-apples on N.
#
# Total wallclock estimate (sequential, T=0.7 sampling):
#   minimax  ~25 min/method ×3 = 1.3h
#   gemini   ~25 min/method ×3 = 1.3h
#   llama3   ~3 h/method   ×3 = 9 h   (Ollama, single 3090, Q4_K_M)
#   judging  ~30 min × 9      = 4.5h
#   ≈ 16h end-to-end on a single 3090 + commercial API access.
#
# Outputs:
#   runs/personalitybench_{mm,gem,l8b}_{caps,sp,p2}/                    (raw)
#   runs/personalitybench_{mm,gem,l8b}_{caps,sp,p2}_scored/summary.json (judged)
#
# Aggregator (after this script finishes):
#   python3 scripts/aggregate_pb_3x3.py  → paper/tables/pb_3x3.md
#
# Reproduces the §4.4 headline 3×3 table to within sampling noise.

set -euo pipefail
export PYTHONUNBUFFERED=1

SUBJECTS=("minimax" "gemini" "llama3")
METHODS=("caps" "sp" "p2")
LIMIT=50
JUDGE="gpt-4o-2024-08-06"
P2_BANK="baselines/NPTI/NPTI/dataset/description.json"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p logs
LOG="logs/pb_3x3_$(date +%Y%m%d-%H%M%S).log"
echo "[run_pb_3backbone_3method] log → $LOG"

case "${OLLAMA_HOST:-}" in
  "") echo "  WARN: OLLAMA_HOST is unset; llama3 runs will hit localhost:11434" ;;
  *)  echo "  ollama host: $OLLAMA_HOST" ;;
esac

map_subject() {
  case "$1" in
    minimax) echo "minimax MiniMax-M2.7-highspeed mm";;
    gemini)  echo "gemini  gemini-2.5-flash         gem";;
    llama3)  echo "ollama  llama3:8b                l8b";;
    *) echo "unknown subject: $1" >&2; exit 2;;
  esac
}

for S in "${SUBJECTS[@]}"; do
  read -r BACKEND MODEL TAG <<<"$(map_subject "$S")"
  for M in "${METHODS[@]}"; do
    OUT="runs/personalitybench_${TAG}_${M}"
    SCORED="${OUT}_scored"
    echo
    echo "=========================================================================="
    echo "[run] subject=$S ($BACKEND/$MODEL)  method=$M  out=$OUT  limit=$LIMIT"
    echo "=========================================================================="

    if [ ! -f "${OUT}/Agreeableness_high.jsonl" ]; then
      EXTRA=""
      if [ "$M" = "p2" ]; then EXTRA="--p2-descriptions $P2_BANK"; fi
      python3 -u experiments/exp_personalitybench.py \
        --backend "$BACKEND" --model "$MODEL" \
        --method "$M" --limit "$LIMIT" \
        --output-dir "$OUT" $EXTRA 2>&1 | tee -a "$LOG"
    else
      echo "  [skip generate] ${OUT} already exists"
    fi

    if [ ! -f "${SCORED}/summary.json" ]; then
      python3 -u experiments/judge_personalitybench.py \
        --input-dir "$OUT" --output-dir "$SCORED" \
        --judge-backend openai --judge-model "$JUDGE" 2>&1 | tee -a "$LOG"
    else
      echo "  [skip judge] ${SCORED}/summary.json already exists"
    fi
  done
done

echo
echo "[run_pb_3backbone_3method] all 9 cells done."
echo "  aggregate → python3 scripts/aggregate_pb_3x3.py"
