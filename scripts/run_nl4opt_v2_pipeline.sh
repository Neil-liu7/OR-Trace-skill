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

TRAIN_FILE="data/benchmarks/NL4OPT_train_converted.jsonl"
TEST_FILE="data/benchmarks/NL4OPT_Test.jsonl"
OUT_DIR="outputs/qwen3_14b_NL4OPT_v2"

mkdir -p "$OUT_DIR"

N_TRAIN=$(wc -l < "$TRAIN_FILE")
N_TEST=$(wc -l < "$TEST_FILE")
echo "=========================================="
echo " Qwen3-14B: NL4OPT v2"
echo " Train: $N_TRAIN items (self-labeled)"
echo " Test: $N_TEST items (with gold answers)"
echo "=========================================="

# Step 1: Think-mode trace generation on training set
if [ ! -f "$OUT_DIR/traces.jsonl" ]; then
  echo "[Step 1] Think-mode trace generation on training set..."
  python3 scripts/2_generate_traces.py \
    "$TRAIN_FILE" prompts/source_cot.txt "$OUT_DIR/traces.jsonl" \
    --config "$CONFIG"
else
  echo "[Step 1] traces.jsonl exists, skipping."
fi

# Step 2: Self-label answers (extract answer from Think-mode output as gold)
# Since train has no gold answers, we treat the model's own Think-mode answer as correct
if [ ! -f "$OUT_DIR/verified_traces.jsonl" ]; then
  echo "[Step 2] Self-labeling answers from traces..."
  python3 -c "
import json, re, sys
sys.path.insert(0, 'src')
from evaluate import extract_answer

traces = [json.loads(l) for l in open('$OUT_DIR/traces.jsonl')]
verified = []
rejected = []

for t in traces:
    response = t.get('response', '')
    # Extract answer from the response
    ans = extract_answer(response)
    if ans is not None:
        t['extracted_answer'] = ans
        t['gold_answer'] = ans  # self-label: model's answer IS the gold
        t['is_correct'] = 1
        verified.append(t)
    else:
        t['is_correct'] = 0
        rejected.append(t)

with open('$OUT_DIR/verified_traces.jsonl', 'w') as f:
    for v in verified:
        f.write(json.dumps(v, ensure_ascii=False) + '\n')

with open('$OUT_DIR/rejected_traces.jsonl', 'w') as f:
    for r in rejected:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f'Verified (answer extracted): {len(verified)}/{len(traces)}')
print(f'Rejected (no answer found): {len(rejected)}/{len(traces)}')
"
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

# Step 4: LOO inference on training set
if [ ! -f "$OUT_DIR/predictions_train_loo.jsonl" ]; then
  echo "[Step 4] Running LOO inference on training set..."
  python3 scripts/5_retrieve_and_infer.py \
    "$TRAIN_FILE" "$OUT_DIR/skills.jsonl" prompts/skill_infer.txt \
    "$OUT_DIR/predictions_train_loo.jsonl" --config "$CONFIG_LOO"
else
  echo "[Step 4] predictions_train_loo.jsonl exists, skipping."
fi

# Step 5: Baseline on training set
if [ ! -f "$OUT_DIR/predictions_train_baseline.jsonl" ]; then
  echo "[Step 5] Running baseline on training set..."
  python3 scripts/baseline_no_think.py \
    "$TRAIN_FILE" "$OUT_DIR/predictions_train_baseline.jsonl" --config "$CONFIG"
else
  echo "[Step 5] predictions_train_baseline.jsonl exists, skipping."
fi

# Step 6: Filter + merge skills
# Note: For self-labeled data, we compare LOO predictions against self-labeled answers
if [ ! -f "$OUT_DIR/skills_filtered.jsonl" ]; then
  echo "[Step 6] Filtering and merging skills..."
  python3 scripts/filter_and_merge_skills.py \
    "$OUT_DIR/skills.jsonl" \
    "$OUT_DIR/predictions_train_loo.jsonl" \
    "$OUT_DIR/predictions_train_baseline.jsonl" \
    "$OUT_DIR/skills_filtered.jsonl" \
    --report "$OUT_DIR/filter_report.json"
else
  echo "[Step 6] skills_filtered.jsonl exists, skipping."
fi

# Step 7: Test + skill (single run, temp=0)
if [ ! -f "$OUT_DIR/predictions_test_skill.jsonl" ]; then
  echo "[Step 7] Running skill inference on test set..."
  python3 scripts/5_retrieve_and_infer.py \
    "$TEST_FILE" "$OUT_DIR/skills_filtered.jsonl" prompts/skill_infer.txt \
    "$OUT_DIR/predictions_test_skill.jsonl" --config "$CONFIG"
else
  echo "[Step 7] predictions_test_skill.jsonl exists, skipping."
fi

# Step 8: Test baseline (single run, temp=0)
if [ ! -f "$OUT_DIR/predictions_test_baseline.jsonl" ]; then
  echo "[Step 8] Running baseline on test set..."
  python3 scripts/baseline_no_think.py \
    "$TEST_FILE" "$OUT_DIR/predictions_test_baseline.jsonl" --config "$CONFIG"
else
  echo "[Step 8] predictions_test_baseline.jsonl exists, skipping."
fi

# Step 9: 5 trials (temp=0.7)
TRIAL_DIR="$OUT_DIR/5trials"
mkdir -p "$TRIAL_DIR"
echo "[Step 9] Running 5-trial evaluation (temperature=0.7)..."
for i in 1 2 3 4 5; do
  echo "  Trial $i/5..."
  if [ ! -f "$TRIAL_DIR/test_skill_trial${i}.jsonl" ]; then
    python3 scripts/5_retrieve_and_infer.py \
      "$TEST_FILE" "$OUT_DIR/skills_filtered.jsonl" prompts/skill_infer.txt \
      "$TRIAL_DIR/test_skill_trial${i}.jsonl" --config "$CONFIG_TEMP"
  fi
  if [ ! -f "$TRIAL_DIR/test_baseline_trial${i}.jsonl" ]; then
    python3 scripts/baseline_no_think.py \
      "$TEST_FILE" "$TRIAL_DIR/test_baseline_trial${i}.jsonl" --config "$CONFIG_TEMP"
  fi
done

# Print results
echo ""
echo "=== NL4OPT v2 (Qwen3-14B) Results ==="
python3 -c "
import json, os
import numpy as np

out_dir = '$OUT_DIR'
trial_dir = '$TRIAL_DIR'

# Single run (temp=0)
for label, path in [('Test+Skill (t=0)', f'{out_dir}/predictions_test_skill.jsonl'),
                    ('Test Baseline (t=0)', f'{out_dir}/predictions_test_baseline.jsonl')]:
    if os.path.exists(path):
        rows = [json.loads(l) for l in open(path)]
        c = sum(1 for r in rows if r.get('is_correct') == 1)
        print(f'  {label}: {c}/{len(rows)} = {c/len(rows)*100:.1f}%')

# 5 trials
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
