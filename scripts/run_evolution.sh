#!/bin/bash
set -e

export HF_HUB_OFFLINE=1

CONFIG="configs/default.yaml"
ROUNDS=3

run_evolution() {
  local BENCH_NAME=$1
  local TRAIN_FILE=$2
  local TEST_FILE=$3
  local SKILL_FILE=$4
  local OUT_DIR="outputs/${BENCH_NAME}_evolved"

  mkdir -p "$OUT_DIR"

  echo "=========================================="
  echo "Lamarckian Evolution: $BENCH_NAME"
  echo "Train:  $TRAIN_FILE ($(wc -l < "$TRAIN_FILE") items)"
  echo "Skills: $SKILL_FILE"
  echo "Rounds: $ROUNDS"
  echo "=========================================="

  # Step 1: Run evolution on training set
  echo "[Step 1] Running ${ROUNDS}-round evolution..."
  python3 scripts/7_evolve_skills.py \
    "$TRAIN_FILE" \
    "$SKILL_FILE" \
    --output-dir "$OUT_DIR/evolution" \
    --config "$CONFIG" \
    --rounds "$ROUNDS"

  EVOLVED_SKILLS="$OUT_DIR/evolution/evolved_skills.jsonl"

  # Step 2: Evaluate evolved skills on test set
  echo ""
  echo "[Step 2] Evaluating evolved skills on test set..."
  python3 scripts/5_retrieve_and_infer.py \
    "$TEST_FILE" \
    "$EVOLVED_SKILLS" \
    prompts/skill_infer.txt \
    "$OUT_DIR/predictions_evolved.jsonl" \
    --config "$CONFIG"

  # Step 3: Evaluate original skills on test set (baseline)
  echo ""
  echo "[Step 3] Evaluating original skills on test set (baseline)..."
  python3 scripts/5_retrieve_and_infer.py \
    "$TEST_FILE" \
    "$SKILL_FILE" \
    prompts/skill_infer.txt \
    "$OUT_DIR/predictions_original.jsonl" \
    --config "$CONFIG"

  # Step 4: Compare
  echo ""
  echo "=== ${BENCH_NAME}: Original vs Evolved ==="
  python3 scripts/6_evaluate.py \
    "$OUT_DIR/predictions_evolved.jsonl" \
    --label "Evolved" \
    --compare "$OUT_DIR/predictions_original.jsonl" \
    --compare-label "Original"
  echo ""
}

# --- Example: MAMO_ComplexLP ---
# Assumes you have already run the base pipeline to produce initial skills.
# Adjust paths to match your existing outputs.

run_evolution \
  "MAMO_ComplexLP_fixed" \
  "data/benchmarks/MAMO_ComplexLP_fixed_train.jsonl" \
  "data/benchmarks/MAMO_ComplexLP_fixed_test.jsonl" \
  "outputs/MAMO_ComplexLP_fixed_full/skills.jsonl"

echo "=========================================="
echo "EVOLUTION COMPLETE"
echo "=========================================="
