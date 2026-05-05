#!/bin/bash
set -e

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export no_proxy="127.0.0.1,localhost"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

CONFIG="configs/qwen3_14b_temp07.yaml"
TEST_FILE="data/splits/MAMO_ComplexLP_test.jsonl"
SKILL_FILE="outputs/qwen3_14b_MAMO_ComplexLP_split/skills_filtered.jsonl"
OUT_DIR="outputs/qwen3_14b_MAMO_ComplexLP_split/5trials"

mkdir -p "$OUT_DIR"

echo "=========================================="
echo " 5-Trial Evaluation: MAMO_ComplexLP Test Set (46 items)"
echo " Temperature: 0.7"
echo "=========================================="

for i in 1 2 3 4 5; do
  echo ""
  echo "--- Trial $i/5 ---"

  # Test + Skill
  echo "  [Skill] Running..."
  python3 scripts/5_retrieve_and_infer.py \
    "$TEST_FILE" \
    "$SKILL_FILE" \
    prompts/skill_infer.txt \
    "$OUT_DIR/test_skill_trial${i}.jsonl" \
    --config "$CONFIG"

  # Baseline
  echo "  [Baseline] Running..."
  python3 scripts/baseline_no_think.py \
    "$TEST_FILE" \
    "$OUT_DIR/test_baseline_trial${i}.jsonl" \
    --config "$CONFIG"
done

# Summary
echo ""
echo "=========================================="
echo " 5-Trial Results Summary"
echo "=========================================="

python3 -c "
import json, os
import numpy as np

out_dir = '$OUT_DIR'
skill_accs = []
base_accs = []

for i in range(1, 6):
    sk_path = f'{out_dir}/test_skill_trial{i}.jsonl'
    bl_path = f'{out_dir}/test_baseline_trial{i}.jsonl'

    if os.path.exists(sk_path):
        rows = [json.loads(l) for l in open(sk_path)]
        c = sum(1 for r in rows if r.get('is_correct') == 1)
        skill_accs.append(c / len(rows) * 100)
    if os.path.exists(bl_path):
        rows = [json.loads(l) for l in open(bl_path)]
        c = sum(1 for r in rows if r.get('is_correct') == 1)
        base_accs.append(c / len(rows) * 100)

print(f'Test + Skill (5 trials):')
for i, a in enumerate(skill_accs, 1):
    print(f'  Trial {i}: {a:.1f}%')
print(f'  Mean: {np.mean(skill_accs):.1f}% ± {np.std(skill_accs):.1f}%')
print()
print(f'Test Baseline (5 trials):')
for i, a in enumerate(base_accs, 1):
    print(f'  Trial {i}: {a:.1f}%')
print(f'  Mean: {np.mean(base_accs):.1f}% ± {np.std(base_accs):.1f}%')
print()
print(f'Delta (Skill - Baseline): {np.mean(skill_accs) - np.mean(base_accs):+.1f}%')
"
