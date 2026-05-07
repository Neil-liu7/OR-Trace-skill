#!/bin/bash
set -e

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export no_proxy="127.0.0.1,localhost"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

CONFIG="configs/qwen3_14b.yaml"
CONFIG_LOO="configs/qwen3_14b_loo.yaml"
CONFIG_TEMP="configs/qwen3_14b_temp07.yaml"

BENCH_NAME="OptMATH_MILP_366"
DATA_FILE="data/benchmarks/OptMATH_MILP_366.jsonl"
OUT_DIR="outputs/qwen3_14b_${BENCH_NAME}"
SPLIT_OUT="outputs/qwen3_14b_${BENCH_NAME}_split"

mkdir -p "$OUT_DIR" "$SPLIT_OUT"

N=$(wc -l < "$DATA_FILE")
echo "=========================================="
echo " Qwen3-14B Pipeline: $BENCH_NAME ($N items)"
echo "=========================================="

# Step 1: Think-mode trace generation
if [ ! -f "$OUT_DIR/traces.jsonl" ]; then
  echo "[Step 1] Think-mode trace generation..."
  python3 scripts/2_generate_traces.py \
    "$DATA_FILE" prompts/source_cot.txt "$OUT_DIR/traces.jsonl" \
    --config "$CONFIG"
else
  echo "[Step 1] traces.jsonl exists, skipping."
fi

# Step 2: Verify answers
if [ ! -f "$OUT_DIR/verified_traces.jsonl" ]; then
  echo "[Step 2] Verifying answers..."
  python3 scripts/3_verify_answers.py \
    "$OUT_DIR/traces.jsonl" "$OUT_DIR/verified_traces.jsonl" \
    --rejected-file "$OUT_DIR/rejected_traces.jsonl"
else
  echo "[Step 2] verified_traces.jsonl exists, skipping."
fi

# Step 3: Distill skills
if [ ! -f "$OUT_DIR/skills.jsonl" ]; then
  echo "[Step 3] Distilling skills..."
  python3 scripts/4_distill_skills.py \
    "$OUT_DIR/verified_traces.jsonl" prompts/skill_distill.txt \
    "$OUT_DIR/skills.jsonl" --config "$CONFIG"
else
  echo "[Step 3] skills.jsonl exists, skipping."
fi

# LOO inference
if [ ! -f "$OUT_DIR/predictions_skill_loo.jsonl" ]; then
  echo "[LOO] Running LOO inference..."
  python3 scripts/5_retrieve_and_infer.py \
    "$DATA_FILE" "$OUT_DIR/skills.jsonl" prompts/skill_infer.txt \
    "$OUT_DIR/predictions_skill_loo.jsonl" --config "$CONFIG_LOO"
else
  echo "[LOO] predictions_skill_loo.jsonl exists, skipping."
fi

# Baseline
if [ ! -f "$OUT_DIR/predictions_nothink.jsonl" ]; then
  echo "[Baseline] Running baseline..."
  python3 scripts/baseline_no_think.py \
    "$DATA_FILE" "$OUT_DIR/predictions_nothink.jsonl" --config "$CONFIG"
else
  echo "[Baseline] predictions_nothink.jsonl exists, skipping."
fi

# Split
if [ ! -f "data/splits/${BENCH_NAME}_train.jsonl" ]; then
  echo "[Split] Splitting based on LOO..."
  python3 scripts/split_train_test.py \
    "$DATA_FILE" "$OUT_DIR/predictions_skill_loo.jsonl" \
    --name "$BENCH_NAME"
else
  echo "[Split] Already split, skipping."
fi

TRAIN_FILE="data/splits/${BENCH_NAME}_train.jsonl"
TEST_FILE="data/splits/${BENCH_NAME}_test.jsonl"
N_TRAIN=$(wc -l < "$TRAIN_FILE")
N_TEST=$(wc -l < "$TEST_FILE")
echo "  Train: $N_TRAIN items, Test: $N_TEST items"

# Extract train-set skills
if [ ! -f "$SPLIT_OUT/skills_raw.jsonl" ]; then
  echo "[Extract] Getting train-set skills..."
  python3 -c "
import json
train_ids = set(json.loads(l)['question_id'] for l in open('$TRAIN_FILE'))
skills = [json.loads(l) for l in open('$OUT_DIR/skills.jsonl')]
train_skills = [s for s in skills if s.get('question_id') in train_ids]
with open('$SPLIT_OUT/skills_raw.jsonl', 'w') as f:
    for s in train_skills:
        f.write(json.dumps(s, ensure_ascii=False) + '\n')
print(f'Extracted {len(train_skills)} train-set skills (from {len(skills)} total)')
"
fi

# Train LOO
if [ ! -f "$SPLIT_OUT/predictions_train_loo.jsonl" ]; then
  echo "[Train LOO] Running LOO on training set..."
  python3 scripts/5_retrieve_and_infer.py \
    "$TRAIN_FILE" "$SPLIT_OUT/skills_raw.jsonl" prompts/skill_infer.txt \
    "$SPLIT_OUT/predictions_train_loo.jsonl" --config "$CONFIG_LOO"
fi

# Train baseline
if [ ! -f "$SPLIT_OUT/predictions_train_baseline.jsonl" ]; then
  echo "[Train Baseline] Running baseline on training set..."
  python3 scripts/baseline_no_think.py \
    "$TRAIN_FILE" "$SPLIT_OUT/predictions_train_baseline.jsonl" --config "$CONFIG"
fi

# Filter + merge
if [ ! -f "$SPLIT_OUT/skills_filtered.jsonl" ]; then
  echo "[Filter] Filtering and merging skills..."
  python3 scripts/filter_and_merge_skills.py \
    "$SPLIT_OUT/skills_raw.jsonl" \
    "$SPLIT_OUT/predictions_train_loo.jsonl" \
    "$SPLIT_OUT/predictions_train_baseline.jsonl" \
    "$SPLIT_OUT/skills_filtered.jsonl" \
    --report "$SPLIT_OUT/filter_report.json"
fi

# Test + skill (temp=0)
if [ ! -f "$SPLIT_OUT/predictions_test_skill.jsonl" ]; then
  echo "[Test Skill] Running skill inference on test set..."
  python3 scripts/5_retrieve_and_infer.py \
    "$TEST_FILE" "$SPLIT_OUT/skills_filtered.jsonl" prompts/skill_infer.txt \
    "$SPLIT_OUT/predictions_test_skill.jsonl" --config "$CONFIG"
fi

# Test baseline (temp=0)
if [ ! -f "$SPLIT_OUT/predictions_test_baseline.jsonl" ]; then
  echo "[Test Baseline] Running baseline on test set..."
  python3 scripts/baseline_no_think.py \
    "$TEST_FILE" "$SPLIT_OUT/predictions_test_baseline.jsonl" --config "$CONFIG"
fi

# 5 trials (temp=0.7)
TRIAL_DIR="$SPLIT_OUT/5trials"
mkdir -p "$TRIAL_DIR"
echo "[5 Trials] Running 5-trial evaluation (temperature=0.7)..."
for i in 1 2 3 4 5; do
  echo "  Trial $i/5..."
  if [ ! -f "$TRIAL_DIR/test_skill_trial${i}.jsonl" ]; then
    python3 scripts/5_retrieve_and_infer.py \
      "$TEST_FILE" "$SPLIT_OUT/skills_filtered.jsonl" prompts/skill_infer.txt \
      "$TRIAL_DIR/test_skill_trial${i}.jsonl" --config "$CONFIG_TEMP"
  fi
  if [ ! -f "$TRIAL_DIR/test_baseline_trial${i}.jsonl" ]; then
    python3 scripts/baseline_no_think.py \
      "$TEST_FILE" "$TRIAL_DIR/test_baseline_trial${i}.jsonl" --config "$CONFIG_TEMP"
  fi
done

# Results
echo ""
echo "=== $BENCH_NAME Results ==="
python3 -c "
import json, os
import numpy as np

split_out = '$SPLIT_OUT'
trial_dir = '$TRIAL_DIR'

for label, path in [('Test+Skill (t=0)', f'{split_out}/predictions_test_skill.jsonl'),
                    ('Test Baseline (t=0)', f'{split_out}/predictions_test_baseline.jsonl')]:
    if os.path.exists(path):
        rows = [json.loads(l) for l in open(path)]
        c = sum(1 for r in rows if r.get('is_correct') == 1)
        print(f'  {label}: {c}/{len(rows)} = {c/len(rows)*100:.1f}%')

skill_accs, base_accs = [], []
for i in range(1, 6):
    sk = f'{trial_dir}/test_skill_trial{i}.jsonl'
    bl = f'{trial_dir}/test_baseline_trial{i}.jsonl'
    if os.path.exists(sk):
        rows = [json.loads(l) for l in open(sk)]
        skill_accs.append(sum(1 for r in rows if r.get('is_correct') == 1) / len(rows) * 100)
    if os.path.exists(bl):
        rows = [json.loads(l) for l in open(bl)]
        base_accs.append(sum(1 for r in rows if r.get('is_correct') == 1) / len(rows) * 100)

if skill_accs:
    print(f'  Skill 5-trial: {np.mean(skill_accs):.1f}% +/- {np.std(skill_accs):.1f}%')
if base_accs:
    print(f'  Baseline 5-trial: {np.mean(base_accs):.1f}% +/- {np.std(base_accs):.1f}%')
if skill_accs and base_accs:
    print(f'  Delta: +{np.mean(skill_accs) - np.mean(base_accs):.1f}%')
"
echo ""
echo "Done!"
