#!/bin/bash
set -e

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export no_proxy="127.0.0.1,localhost"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

CONFIG="configs/qwen3_4b.yaml"
CONFIG_LOO="configs/qwen3_4b_loo.yaml"
CONFIG_TEMP="configs/qwen3_4b_temp07.yaml"

# ==========================================
# Full pipeline for a benchmark (4B)
# Uses pre-existing train/test splits from 14B
# ==========================================
run_full_pipeline() {
  local BENCH_NAME=$1
  local TRAIN_FILE=$2
  local TEST_FILE=$3
  local OUT_DIR="outputs/qwen3_4b_${BENCH_NAME}"

  mkdir -p "$OUT_DIR"

  local N_TRAIN=$(wc -l < "$TRAIN_FILE")
  local N_TEST=$(wc -l < "$TEST_FILE")
  echo ""
  echo "=========================================="
  echo " Qwen3-4B Pipeline: $BENCH_NAME"
  echo " Train: $N_TRAIN, Test: $N_TEST"
  echo "=========================================="

  # Step 1: Think-mode trace generation (on training set)
  if [ ! -f "$OUT_DIR/traces.jsonl" ]; then
    echo "[Step 1] Think-mode trace generation on training set..."
    python3 scripts/2_generate_traces.py \
      "$TRAIN_FILE" prompts/source_cot.txt "$OUT_DIR/traces.jsonl" \
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
  local TRIAL_DIR="$OUT_DIR/5trials"
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
  echo "=== $BENCH_NAME (Qwen3-4B) Results ==="
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
}


# ==========================================
# Main: Run all benchmarks for Qwen3-4B
# Uses the same train/test splits as Qwen3-14B
# ==========================================

echo "################################################################"
echo "# Qwen3-4B Full Pipeline (using 14B splits)"
echo "################################################################"

# 1. ComplexLP
echo "################################################################"
echo "# 1. MAMO_ComplexLP (Train:157, Test:46)"
echo "################################################################"
run_full_pipeline "MAMO_ComplexLP" \
  "data/splits/MAMO_ComplexLP_train.jsonl" \
  "data/splits/MAMO_ComplexLP_test.jsonl"

# 2. EasyLP
echo "################################################################"
echo "# 2. MAMO_EasyLP (Train:454, Test:188)"
echo "################################################################"
run_full_pipeline "MAMO_EasyLP" \
  "data/splits/MAMO_EasyLP_train.jsonl" \
  "data/splits/MAMO_EasyLP_test.jsonl"

# 3. IndustryOR
echo "################################################################"
echo "# 3. IndustryOR (Train:84, Test:16)"
echo "################################################################"
run_full_pipeline "IndustryOR" \
  "data/splits/IndustryOR_train.jsonl" \
  "data/splits/IndustryOR_test.jsonl"

# 4. NL4OPT
echo "################################################################"
echo "# 4. NL4OPT (Train:176, Test:69)"
echo "################################################################"
run_full_pipeline "NL4OPT" \
  "data/splits/NL4OPT_train.jsonl" \
  "data/splits/NL4OPT_test.jsonl"

# 5. OptMATH_MILP_366
echo "################################################################"
echo "# 5. OptMATH_MILP_366"
echo "################################################################"
if [ -f "data/splits/OptMATH_MILP_366_train.jsonl" ] && [ -f "data/splits/OptMATH_MILP_366_test.jsonl" ]; then
  run_full_pipeline "OptMATH_MILP_366" \
    "data/splits/OptMATH_MILP_366_train.jsonl" \
    "data/splits/OptMATH_MILP_366_test.jsonl"
else
  echo "  [SKIP] OptMATH_MILP_366 splits not yet available (14B pipeline still running)"
fi

# 6. OptiBench
echo "################################################################"
echo "# 6. OptiBench"
echo "################################################################"
if [ -f "data/splits/OptiBench_train.jsonl" ] && [ -f "data/splits/OptiBench_test.jsonl" ]; then
  run_full_pipeline "OptiBench" \
    "data/splits/OptiBench_train.jsonl" \
    "data/splits/OptiBench_test.jsonl"
else
  echo "  [SKIP] OptiBench splits not yet available (14B pipeline still running)"
fi

# ==========================================
# Final Summary
# ==========================================
echo ""
echo "################################################################"
echo "# FINAL SUMMARY: Qwen3-4B All Benchmarks"
echo "################################################################"
python3 -c "
import json, os
import numpy as np

benchmarks = ['MAMO_ComplexLP', 'MAMO_EasyLP', 'IndustryOR', 'NL4OPT', 'OptMATH_MILP_366', 'OptiBench']

print(f'{\"Benchmark\":<20} {\"N_test\":>6} {\"Skill(mean)\":>12} {\"Base(mean)\":>12} {\"Delta\":>8}')
print('-' * 65)

for name in benchmarks:
    trial_dir = f'outputs/qwen3_4b_{name}/5trials'

    # Get test count from test file
    test_file = f'data/splits/{name}_test.jsonl'
    n_test = '?'
    if os.path.exists(test_file):
        n_test = sum(1 for _ in open(test_file))

    skill_accs, base_accs = [], []
    for i in range(1, 6):
        sk = f'{trial_dir}/test_skill_trial{i}.jsonl'
        bl = f'{trial_dir}/test_baseline_trial{i}.jsonl'
        if os.path.exists(sk):
            rows = [json.loads(l) for l in open(sk)]
            skill_accs.append(sum(1 for r in rows if r.get('is_correct') == 1) / max(len(rows),1) * 100)
        if os.path.exists(bl):
            rows = [json.loads(l) for l in open(bl)]
            base_accs.append(sum(1 for r in rows if r.get('is_correct') == 1) / max(len(rows),1) * 100)

    if skill_accs and base_accs:
        sm = np.mean(skill_accs)
        bm = np.mean(base_accs)
        delta = sm - bm
        print(f'{name:<20} {n_test:>6} {sm:>10.1f}% {bm:>10.1f}% {delta:>+7.1f}%')
    else:
        print(f'{name:<20} {n_test:>6} {\"N/A\":>12} {\"N/A\":>12} {\"N/A\":>8}')

print()
print('Done!')
"
