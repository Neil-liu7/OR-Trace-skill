#!/bin/bash
set -e

BENCH_NAME=$1
TRAIN_FILE="data/benchmarks/${BENCH_NAME}_train.jsonl"
TEST_FILE="data/benchmarks/${BENCH_NAME}_test.jsonl"
OUT_DIR="outputs/${BENCH_NAME}"
CONFIG="configs/default.yaml"

if [ -z "$BENCH_NAME" ]; then
  echo "Usage: $0 <benchmark_name>"
  exit 1
fi

export HF_HUB_OFFLINE=1

mkdir -p "$OUT_DIR"

echo "=========================================="
echo "Benchmark: $BENCH_NAME"
echo "Train: $TRAIN_FILE"
echo "Test:  $TEST_FILE"
echo "=========================================="

# Step 2: Generate traces (Think mode on train set)
echo "[Step 2] Generating traces..."
python3 scripts/2_generate_traces.py "$TRAIN_FILE" prompts/source_cot.txt "$OUT_DIR/traces.jsonl" --config "$CONFIG"

# Step 3: Verify answers
echo "[Step 3] Verifying traces..."
python3 scripts/3_verify_answers.py "$OUT_DIR/traces.jsonl" "$OUT_DIR/verified_traces.jsonl"

# Step 4: Distill skills
echo "[Step 4] Distilling skills..."
python3 scripts/4_distill_skills.py "$OUT_DIR/verified_traces.jsonl" prompts/skill_distill.txt "$OUT_DIR/skills.jsonl" --config "$CONFIG"

# Step 5: Retrieve + Infer (Structural Hybrid, No-Think)
echo "[Step 5] Skill retrieval + inference..."
python3 scripts/5_retrieve_and_infer.py "$TEST_FILE" "$OUT_DIR/skills.jsonl" prompts/skill_infer.txt "$OUT_DIR/predictions_skill.jsonl" --config "$CONFIG"

# Baseline: No-Think, no skill
echo "[Baseline] No-Think, no skill..."
python3 scripts/baseline_no_think.py "$TEST_FILE" "$OUT_DIR/predictions_nothink.jsonl" --config "$CONFIG"

# Baseline: Think mode, 16384 tokens
echo "[Baseline] Think mode, 16384 tokens..."
python3 -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['inference']['no_think'] = False
cfg['inference']['max_tokens'] = 16384
with open('$OUT_DIR/config_think16k.yaml', 'w') as f:
    yaml.dump(cfg, f)
"
python3 scripts/baseline_no_think.py "$TEST_FILE" "$OUT_DIR/predictions_think16k.jsonl" --config "$OUT_DIR/config_think16k.yaml"

echo ""
echo "=========================================="
echo "Results for $BENCH_NAME:"
echo "=========================================="
echo "Skill (structural_hybrid):"
python3 -c "
import json
rows=[json.loads(l) for l in open('$OUT_DIR/predictions_skill.jsonl')]
c=sum(1 for r in rows if r.get('is_correct')==1)
print(f'  {c}/{len(rows)} = {c/len(rows)*100:.1f}%')
"
echo "No-Think baseline:"
python3 -c "
import json
rows=[json.loads(l) for l in open('$OUT_DIR/predictions_nothink.jsonl')]
c=sum(1 for r in rows if r.get('is_correct')==1)
print(f'  {c}/{len(rows)} = {c/len(rows)*100:.1f}%')
"
echo "Think 16k baseline:"
python3 -c "
import json
rows=[json.loads(l) for l in open('$OUT_DIR/predictions_think16k.jsonl')]
c=sum(1 for r in rows if r.get('is_correct')==1)
print(f'  {c}/{len(rows)} = {c/len(rows)*100:.1f}%')
"
