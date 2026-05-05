#!/bin/bash
set -e

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export no_proxy="127.0.0.1,localhost"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

CONFIG="configs/qwen3_14b.yaml"

# ==========================================
# Full Pipeline: Train Split → Skills → Filter → Test Eval
# ==========================================

run_split_pipeline() {
  local BENCH_NAME=$1
  local TRAIN_FILE="data/splits/${BENCH_NAME}_train.jsonl"
  local TEST_FILE="data/splits/${BENCH_NAME}_test.jsonl"
  local OUT_DIR="outputs/qwen3_14b_${BENCH_NAME}_split"

  mkdir -p "$OUT_DIR"

  local N_TRAIN=$(wc -l < "$TRAIN_FILE")
  local N_TEST=$(wc -l < "$TEST_FILE")
  echo ""
  echo "=========================================="
  echo "Pipeline: $BENCH_NAME"
  echo "Train: $TRAIN_FILE ($N_TRAIN items)"
  echo "Test:  $TEST_FILE ($N_TEST items)"
  echo "Output: $OUT_DIR"
  echo "=========================================="

  # Step 1: Think-mode trace generation on TRAINING set
  echo "[Step 1] Think-mode trace generation on training set..."
  python3 scripts/2_generate_traces.py \
    "$TRAIN_FILE" \
    prompts/source_cot.txt \
    "$OUT_DIR/traces.jsonl" \
    --config "$CONFIG"

  # Step 2: Verify answers
  echo "[Step 2] Verifying answers..."
  python3 scripts/3_verify_answers.py \
    "$OUT_DIR/traces.jsonl" \
    "$OUT_DIR/verified_traces.jsonl" \
    --rejected-file "$OUT_DIR/rejected_traces.jsonl"

  # Step 3: Distill skills from correct traces
  echo "[Step 3] Distilling skills..."
  python3 scripts/4_distill_skills.py \
    "$OUT_DIR/verified_traces.jsonl" \
    prompts/skill_distill.txt \
    "$OUT_DIR/skills_raw.jsonl" \
    --config "$CONFIG"

  # Step 4: Filter and merge skills (using training-set LOO for metrics)
  # We need LOO predictions on training set for filtering
  # Run LOO on training set first
  echo "[Step 4a] Running LOO inference on training set for skill metrics..."
  python3 -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['inference']['exclude_self'] = True
with open('$OUT_DIR/config_loo.yaml', 'w') as f:
    yaml.dump(cfg, f)
"
  python3 scripts/5_retrieve_and_infer.py \
    "$TRAIN_FILE" \
    "$OUT_DIR/skills_raw.jsonl" \
    prompts/skill_infer.txt \
    "$OUT_DIR/predictions_train_loo.jsonl" \
    --config "$OUT_DIR/config_loo.yaml"

  # Baseline on training set
  echo "[Step 4b] Baseline on training set..."
  python3 scripts/baseline_no_think.py \
    "$TRAIN_FILE" \
    "$OUT_DIR/predictions_train_baseline.jsonl" \
    --config "$CONFIG"

  # Filter and merge
  echo "[Step 4c] Filtering and merging skills..."
  python3 scripts/filter_and_merge_skills.py \
    "$OUT_DIR/skills_raw.jsonl" \
    "$OUT_DIR/predictions_train_loo.jsonl" \
    "$OUT_DIR/predictions_train_baseline.jsonl" \
    "$OUT_DIR/skills_filtered.jsonl" \
    --report "$OUT_DIR/filter_report.json"

  # Step 5: Retrieve + NoThink inference on TEST set using filtered skills
  echo "[Step 5] Skill retrieval + NoThink inference on test set..."
  python3 scripts/5_retrieve_and_infer.py \
    "$TEST_FILE" \
    "$OUT_DIR/skills_filtered.jsonl" \
    prompts/skill_infer.txt \
    "$OUT_DIR/predictions_test_skill.jsonl" \
    --config "$CONFIG"

  # Step 6: Baseline on test set
  echo "[Step 6] Baseline NoThink on test set..."
  python3 scripts/baseline_no_think.py \
    "$TEST_FILE" \
    "$OUT_DIR/predictions_test_baseline.jsonl" \
    --config "$CONFIG"

  # Print results
  echo ""
  echo "=== ${BENCH_NAME} Split Pipeline Results ==="
  python3 -c "
import json, os
results = {}
for label, path in [
    ('Test+Skill', '$OUT_DIR/predictions_test_skill.jsonl'),
    ('Test Baseline', '$OUT_DIR/predictions_test_baseline.jsonl'),
    ('Train LOO', '$OUT_DIR/predictions_train_loo.jsonl'),
    ('Train Baseline', '$OUT_DIR/predictions_train_baseline.jsonl'),
]:
    if os.path.exists(path):
        rows = [json.loads(l) for l in open(path)]
        c = sum(1 for r in rows if r.get('is_correct') == 1)
        n = len(rows)
        print(f'  {label}: {c}/{n} = {c/n*100:.1f}%')
    else:
        print(f'  {label}: NOT FOUND')

# Skill stats
sk_raw = '$OUT_DIR/skills_raw.jsonl'
sk_filt = '$OUT_DIR/skills_filtered.jsonl'
if os.path.exists(sk_raw):
    raw = [json.loads(l) for l in open(sk_raw)]
    raw_ok = sum(1 for s in raw if s.get('status') == 'success')
    print(f'  Skills (raw): {raw_ok}/{len(raw)}')
if os.path.exists(sk_filt):
    filt = [json.loads(l) for l in open(sk_filt)]
    print(f'  Skills (filtered+merged): {len(filt)}')
"
  echo ""
}

# Run on ComplexLP
run_split_pipeline "MAMO_ComplexLP"

# Run on EasyLP
run_split_pipeline "MAMO_EasyLP"

echo "=========================================="
echo " ALL DONE"
echo "=========================================="
