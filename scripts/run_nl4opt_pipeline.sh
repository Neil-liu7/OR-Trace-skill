#!/bin/bash
set -e

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export no_proxy="127.0.0.1,localhost"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

CONFIG="configs/nl4opt_14b.yaml"
TRAIN_FILE="data/train_data/NL4OPT_Train.jsonl"
TEST_FILE="data/benchmarks/NL4OPT_Test.jsonl"
OUT_DIR="outputs/nl4opt_14b"

mkdir -p "$OUT_DIR"

echo "=========================================="
echo " NL4OPT Self-Evolution Pipeline (Qwen3-14B)"
echo " Train: $TRAIN_FILE ($(wc -l < "$TRAIN_FILE") items)"
echo " Test:  $TEST_FILE ($(wc -l < "$TEST_FILE") items)"
echo "=========================================="

# Step 1: Think-mode trace generation
if [ ! -f "$OUT_DIR/traces.jsonl" ]; then
  echo ""
  echo "[Step 1] Generating Think-mode traces on training set..."
  python3 scripts/2_generate_traces.py \
    "$TRAIN_FILE" prompts/source_cot.txt \
    "$OUT_DIR/traces.jsonl" --config "$CONFIG"
else
  echo "[Step 1] SKIP — traces.jsonl already exists"
fi

# Step 2: Verify traces
if [ ! -f "$OUT_DIR/verified_traces.jsonl" ]; then
  echo ""
  echo "[Step 2] Verifying traces..."
  python3 scripts/3_verify_answers.py \
    "$OUT_DIR/traces.jsonl" "$OUT_DIR/verified_traces.jsonl"
else
  echo "[Step 2] SKIP — verified_traces.jsonl already exists"
fi

# Step 3: Distill skills
if [ ! -f "$OUT_DIR/skills_raw.jsonl" ]; then
  echo ""
  echo "[Step 3] Distilling skills from verified traces..."
  python3 scripts/4_distill_skills.py \
    "$OUT_DIR/verified_traces.jsonl" prompts/skill_distill.txt \
    "$OUT_DIR/skills_raw.jsonl" --config "$CONFIG"
else
  echo "[Step 3] SKIP — skills_raw.jsonl already exists"
fi

# Step 4a: LOO inference on training set
if [ ! -f "$OUT_DIR/predictions_train_loo.jsonl" ]; then
  echo ""
  echo "[Step 4a] LOO inference on training set..."
  python3 scripts/5_retrieve_and_infer.py \
    "$TRAIN_FILE" "$OUT_DIR/skills_raw.jsonl" \
    prompts/skill_infer.txt "$OUT_DIR/predictions_train_loo.jsonl" \
    --config "$CONFIG" --exclude-self
else
  echo "[Step 4a] SKIP — predictions_train_loo.jsonl already exists"
fi

# Step 4b: Baseline inference on training set
if [ ! -f "$OUT_DIR/predictions_train_baseline.jsonl" ]; then
  echo ""
  echo "[Step 4b] Baseline inference on training set..."
  python3 scripts/baseline_no_think.py \
    "$TRAIN_FILE" "$OUT_DIR/predictions_train_baseline.jsonl" \
    --config "$CONFIG"
else
  echo "[Step 4b] SKIP — predictions_train_baseline.jsonl already exists"
fi

# Step 5: Filter and merge skills
if [ ! -f "$OUT_DIR/skills_filtered.jsonl" ]; then
  echo ""
  echo "[Step 5] Filtering and merging skills..."
  python3 scripts/filter_and_merge_skills.py \
    "$OUT_DIR/skills_raw.jsonl" \
    "$OUT_DIR/predictions_train_loo.jsonl" \
    "$OUT_DIR/predictions_train_baseline.jsonl" \
    "$OUT_DIR/skills_filtered.jsonl" \
    --report "$OUT_DIR/filter_report.json"
else
  echo "[Step 5] SKIP — skills_filtered.jsonl already exists"
fi

# Step 6a: Test with filtered skills
echo ""
echo "[Step 6a] Test inference with filtered skills..."
python3 scripts/5_retrieve_and_infer.py \
  "$TEST_FILE" "$OUT_DIR/skills_filtered.jsonl" \
  prompts/skill_infer.txt "$OUT_DIR/predictions_test_skill.jsonl" \
  --config "$CONFIG"

# Step 6b: Test baseline
echo ""
echo "[Step 6b] Test baseline (no skill)..."
python3 scripts/baseline_no_think.py \
  "$TEST_FILE" "$OUT_DIR/predictions_test_baseline.jsonl" \
  --config "$CONFIG"

# Summary
echo ""
echo "=========================================="
echo " RESULTS"
echo "=========================================="
python3 -c "
import json, sys
sys.path.insert(0, 'src')
from data import answers_match

for label, path in [
    ('Test+Skill', '$OUT_DIR/predictions_test_skill.jsonl'),
    ('Test Baseline', '$OUT_DIR/predictions_test_baseline.jsonl'),
]:
    rows = [json.loads(l) for l in open(path)]
    c = sum(1 for r in rows if answers_match(r.get('predicted_answer',''), str(r.get('answer',''))))
    print(f'  {label}: {c}/{len(rows)} = {c/len(rows)*100:.1f}%')

# Skill stats
import os
if os.path.exists('$OUT_DIR/verified_traces.jsonl'):
    vt = sum(1 for _ in open('$OUT_DIR/verified_traces.jsonl'))
    print(f'  Verified traces: {vt}/711')
if os.path.exists('$OUT_DIR/skills_raw.jsonl'):
    sr = sum(1 for _ in open('$OUT_DIR/skills_raw.jsonl'))
    print(f'  Skills raw: {sr}')
if os.path.exists('$OUT_DIR/skills_filtered.jsonl'):
    sf = sum(1 for _ in open('$OUT_DIR/skills_filtered.jsonl'))
    print(f'  Skills filtered: {sf}')
"
echo ""
echo "DONE"
