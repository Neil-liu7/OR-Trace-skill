#!/bin/bash
set -e

export HF_HUB_OFFLINE=1
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
CONFIG="configs/default.yaml"

run_one() {
  local BENCH_NAME=$1
  local TRAIN_FILE="data/benchmarks/${BENCH_NAME}_train.jsonl"
  local TEST_FILE="data/benchmarks/${BENCH_NAME}_test.jsonl"
  local OUT_DIR="outputs/${BENCH_NAME}_v2"

  mkdir -p "$OUT_DIR"

  echo ""
  echo "=========================================="
  echo "Benchmark: $BENCH_NAME"
  echo "Train: $TRAIN_FILE ($(wc -l < $TRAIN_FILE) items)"
  echo "Test:  $TEST_FILE ($(wc -l < $TEST_FILE) items)"
  echo "Output: $OUT_DIR"
  echo "=========================================="

  # Step 2: Generate traces (Think mode on train set)
  echo "[Step 2] Generating traces..."
  python3 scripts/2_generate_traces.py "$TRAIN_FILE" prompts/source_cot.txt "$OUT_DIR/traces.jsonl" --config "$CONFIG"

  # Step 3: Verify answers
  echo "[Step 3] Verifying traces..."
  python3 scripts/3_verify_answers.py "$OUT_DIR/traces.jsonl" "$OUT_DIR/verified_traces.jsonl" --rejected-file "$OUT_DIR/rejected_traces.jsonl"

  # Step 3.5: Distill failure skills from rejected traces
  echo "[Step 3.5] Distilling failure skills..."
  python3 scripts/3.5_distill_failure_skills.py "$OUT_DIR/rejected_traces.jsonl" prompts/failure_skill_distill.txt "$OUT_DIR/failure_skills.jsonl" --config "$CONFIG"

  # Step 4: Distill success skills
  echo "[Step 4] Distilling success skills..."
  python3 scripts/4_distill_skills.py "$OUT_DIR/verified_traces.jsonl" prompts/skill_distill.txt "$OUT_DIR/skills.jsonl" --config "$CONFIG"

  # Step 5: Retrieve + Infer (No-Think + Skill Bank with failure skills)
  echo "[Step 5] Skill retrieval + No-Think inference..."
  python3 scripts/5_retrieve_and_infer.py "$TEST_FILE" "$OUT_DIR/skills.jsonl" prompts/skill_infer.txt "$OUT_DIR/predictions_skill.jsonl" --failure-skill-file "$OUT_DIR/failure_skills.jsonl" --config "$CONFIG"

  # Baseline: No-Think, no skill
  echo "[Baseline] No-Think, no skill..."
  python3 scripts/baseline_no_think.py "$TEST_FILE" "$OUT_DIR/predictions_nothink.jsonl" --config "$CONFIG"

  # Print results
  echo ""
  echo "=== ${BENCH_NAME} Results ==="
  python3 -c "
import json, sys
sys.path.insert(0, 'src')
from data import answers_match

for label, fname in [('NoThink+SkillBank', 'predictions_skill.jsonl'), ('NoThink(baseline)', 'predictions_nothink.jsonl')]:
    path = '$OUT_DIR/' + fname
    rows = [json.loads(l) for l in open(path)]
    c = sum(1 for r in rows if r.get('is_correct') == 1)
    print(f'  {label}: {c}/{len(rows)} = {c/len(rows)*100:.1f}%')

# Skill stats
import os
vt = [json.loads(l) for l in open('$OUT_DIR/verified_traces.jsonl')]
sk = [json.loads(l) for l in open('$OUT_DIR/skills.jsonl')]
fsk_path = '$OUT_DIR/failure_skills.jsonl'
fsk = [json.loads(l) for l in open(fsk_path)] if os.path.exists(fsk_path) else []
fsk_ok = [s for s in fsk if s.get('status') == 'success']
print(f'  Verified traces: {len(vt)}')
print(f'  Success skills: {len(sk)}')
print(f'  Failure skills: {len(fsk_ok)} (from {len(fsk)} rejected)')

# Retrieval mode breakdown
skill_rows = [json.loads(l) for l in open('$OUT_DIR/predictions_skill.jsonl')]
from collections import Counter
modes = Counter(r.get('retrieval_mode', 'unknown') for r in skill_rows)
print(f'  Retrieval modes: {dict(modes)}')
"
  echo ""
}

run_one "MAMO_ComplexLP_fixed"
run_one "IndustryOR_fixedV2"
run_one "OptMATH_Bench_166"

echo ""
echo "=========================================="
echo "ALL BENCHMARKS COMPLETE"
echo "=========================================="
echo ""

# Final summary table
python3 -c "
import json, os

benchmarks = ['MAMO_ComplexLP_fixed', 'IndustryOR_fixedV2', 'OptMATH_Bench_166']
print(f'{'Benchmark':<25} {'NoThink+Skill':>15} {'NoThink(base)':>15} {'Delta':>10}')
print('-' * 68)
for b in benchmarks:
    d = f'outputs/{b}_v2'
    results = {}
    for label, fname in [('skill', 'predictions_skill.jsonl'), ('base', 'predictions_nothink.jsonl')]:
        path = f'{d}/{fname}'
        if os.path.exists(path):
            rows = [json.loads(l) for l in open(path)]
            c = sum(1 for r in rows if r.get('is_correct') == 1)
            results[label] = (c, len(rows))
        else:
            results[label] = (0, 0)
    sc, sn = results['skill']
    bc, bn = results['base']
    sp = sc/sn*100 if sn else 0
    bp = bc/bn*100 if bn else 0
    delta = sp - bp
    print(f'{b:<25} {sc}/{sn} ({sp:.1f}%){\"\":>3} {bc}/{bn} ({bp:.1f}%){\"\":>3} {delta:+.1f}%')
"
