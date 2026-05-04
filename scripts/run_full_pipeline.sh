#!/bin/bash
set -e

export HF_HUB_OFFLINE=1

run_benchmark() {
  local BENCH_NAME=$1
  local DATA_FILE=$2
  local OUT_DIR="outputs/${BENCH_NAME}_full"
  local CONFIG="configs/default.yaml"

  mkdir -p "$OUT_DIR"

  echo "=========================================="
  echo "Benchmark: $BENCH_NAME (full, $(wc -l < "$DATA_FILE") items)"
  echo "Data:  $DATA_FILE"
  echo "Output: $OUT_DIR"
  echo "=========================================="

  # --- Think config ---
  python3 -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['inference']['no_think'] = False
cfg['inference']['max_tokens'] = 16384
with open('$OUT_DIR/config_think16k.yaml', 'w') as f:
    yaml.dump(cfg, f)
"

  # Step 1: Think 16k 解题 (同时作为 baseline 和 trace 来源)
  echo "[Step 1] Think 16k inference (baseline + trace source)..."
  python3 scripts/baseline_no_think.py "$DATA_FILE" "$OUT_DIR/predictions_think16k.jsonl" --config "$OUT_DIR/config_think16k.yaml"

  # Step 2: 从 Think 结果中筛选正确答案
  echo "[Step 2] Verifying Think results..."
  python3 scripts/3_verify_answers.py "$OUT_DIR/predictions_think16k.jsonl" "$OUT_DIR/verified_traces.jsonl"

  # Step 3: 从正确的 thinking 过程提取 skill
  echo "[Step 3] Distilling skills from correct thinking traces..."
  python3 scripts/4_distill_skills.py "$OUT_DIR/verified_traces.jsonl" prompts/skill_distill.txt "$OUT_DIR/skills.jsonl" --config "$CONFIG"

  # Step 4: Skill 检索 + No-Think 推理
  echo "[Step 4] Skill retrieval + No-Think inference..."
  python3 scripts/5_retrieve_and_infer.py "$DATA_FILE" "$OUT_DIR/skills.jsonl" prompts/skill_infer.txt "$OUT_DIR/predictions_skill.jsonl" --config "$CONFIG"

  # Step 5: No-Think baseline (无 skill)
  echo "[Step 5] No-Think baseline (no skill)..."
  python3 scripts/baseline_no_think.py "$DATA_FILE" "$OUT_DIR/predictions_nothink.jsonl" --config "$CONFIG"

  echo ""
  echo "=== ${BENCH_NAME} Results ==="
  python3 -c "
import json, sys
sys.path.insert(0, 'src')
from data import answers_match
for label, fname in [('Think16k', 'predictions_think16k.jsonl'), ('Skill(NoThink)', 'predictions_skill.jsonl'), ('NoThink', 'predictions_nothink.jsonl')]:
    rows = [json.loads(l) for l in open('$OUT_DIR/' + fname)]
    c = sum(1 for r in rows if answers_match(r.get('predicted_answer',''), r.get('answer','')))
    print(f'  {label}: {c}/{len(rows)} = {c/len(rows)*100:.1f}%')
vt = [json.loads(l) for l in open('$OUT_DIR/verified_traces.jsonl')]
sk = [json.loads(l) for l in open('$OUT_DIR/skills.jsonl')]
print(f'  Verified traces: {len(vt)}/{len(rows)}')
print(f'  Skills extracted: {len(sk)}')
"
  echo ""
}

run_benchmark "MAMO_ComplexLP_fixed" "data/benchmarks/MAMO_ComplexLP_fixed.jsonl"
run_benchmark "IndustryOR_fixedV2" "data/benchmarks/IndustryOR_fixedV2.jsonl"
run_benchmark "OptMATH_Bench_166" "data/benchmarks/OptMATH_Bench_166.jsonl"

echo "=========================================="
echo "ALL BENCHMARKS COMPLETE"
echo "=========================================="
