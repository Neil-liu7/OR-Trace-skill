#!/bin/bash
set -e

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export no_proxy="127.0.0.1,localhost"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

CONFIG="configs/qwen3_14b.yaml"
MODEL_PATH="/root/paddlejob/workspace/liuaofan/OR-Trace-Skill/model/Qwen3-14B"

# ==========================================
# Part 0: Verify vLLM server is running
# ==========================================
echo "Checking vLLM server on port 8001..."
if ! curl -s --noproxy '*' http://127.0.0.1:8001/v1/models | grep -q "data"; then
  echo "ERROR: vLLM not running on port 8001. Start it manually first:"
  echo "  CUDA_VISIBLE_DEVICES=3,4 HF_HUB_OFFLINE=1 python3 -m vllm.entrypoints.openai.api_server \\"
  echo "    --model $MODEL_PATH --port 8001 --tensor-parallel-size 2 --max-model-len 32768 \\"
  echo "    --trust-remote-code --enable-reasoning --reasoning-parser deepseek_r1 --gpu-memory-utilization 0.6"
  exit 1
fi
echo "vLLM is ready."

# ==========================================
# Part 1: Self-Evolution Pipeline (full data)
# ==========================================

run_full_pipeline() {
  local BENCH_NAME=$1
  local DATA_FILE=$2
  local OUT_DIR="outputs/qwen3_14b_${BENCH_NAME}"

  mkdir -p "$OUT_DIR"

  local N=$(wc -l < "$DATA_FILE")
  echo ""
  echo "=========================================="
  echo "Self-Evolution: $BENCH_NAME ($N items)"
  echo "Data:   $DATA_FILE"
  echo "Output: $OUT_DIR"
  echo "=========================================="

  # --- Think config (for trace generation) ---
  python3 -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['inference']['no_think'] = False
cfg['inference']['max_tokens'] = 16384
with open('$OUT_DIR/config_think.yaml', 'w') as f:
    yaml.dump(cfg, f)
"

  # Step 1: Think-mode trace generation
  echo "[Step 1] Think-mode trace generation..."
  python3 scripts/2_generate_traces.py \
    "$DATA_FILE" \
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
    "$OUT_DIR/skills.jsonl" \
    --config "$CONFIG"

  # Step 4: Skill retrieval + No-Think inference
  echo "[Step 4] Skill retrieval + No-Think inference..."
  python3 scripts/5_retrieve_and_infer.py \
    "$DATA_FILE" \
    "$OUT_DIR/skills.jsonl" \
    prompts/skill_infer.txt \
    "$OUT_DIR/predictions_skill.jsonl" \
    --config "$CONFIG"

  # Step 5: No-Think baseline (no skill)
  echo "[Step 5] No-Think baseline..."
  python3 scripts/baseline_no_think.py \
    "$DATA_FILE" \
    "$OUT_DIR/predictions_nothink.jsonl" \
    --config "$CONFIG"

  # Print results
  echo ""
  echo "=== ${BENCH_NAME} Results ==="
  python3 -c "
import json, sys, os
sys.path.insert(0, 'src')
from data import answers_match

for label, fname in [('Think(trace)', 'traces.jsonl'), ('NoThink+Skill', 'predictions_skill.jsonl'), ('NoThink(base)', 'predictions_nothink.jsonl')]:
    path = '$OUT_DIR/' + fname
    if not os.path.exists(path):
        continue
    rows = [json.loads(l) for l in open(path)]
    if label == 'Think(trace)':
        from data import extract_final_answer
        c = 0
        for r in rows:
            if r.get('status') != 'success':
                continue
            resp = r.get('model_response', '') or r.get('raw_model_response', '')
            pred = extract_final_answer(resp)
            if not pred:
                think = r.get('model_think', '') or r.get('raw_reasoning', '')
                pred = extract_final_answer(think)
            if answers_match(pred, r.get('answer', '')):
                c += 1
    else:
        c = sum(1 for r in rows if r.get('is_correct') == 1)
    n = len(rows)
    print(f'  {label}: {c}/{n} = {c/n*100:.1f}%')

vt = [json.loads(l) for l in open('$OUT_DIR/verified_traces.jsonl')]
sk = [json.loads(l) for l in open('$OUT_DIR/skills.jsonl')]
sk_ok = [s for s in sk if s.get('status') == 'success']
print(f'  Verified traces: {len(vt)}')
print(f'  Skills extracted: {len(sk_ok)}/{len(sk)}')
"
  echo ""
}

# Run on all benchmarks (full data, no split)
run_full_pipeline "MAMO_ComplexLP"   "data/benchmarks/MAMO_ComplexLP_fixed.jsonl"
run_full_pipeline "MAMO_EasyLP"      "data/benchmarks/MAMO_EasyLP_fixed.jsonl"
run_full_pipeline "IndustryOR"       "data/benchmarks/IndustryOR_fixedV2.jsonl"
run_full_pipeline "OptMATH_166"      "data/benchmarks/OptMATH_Bench_166.jsonl"
run_full_pipeline "NL4OPT"           "data/benchmarks/NL4OPT.jsonl"
run_full_pipeline "OptiBench"        "data/benchmarks/OptiBench.jsonl"
run_full_pipeline "MILP_eval"        "data/benchmarks/MILP_all_eval.jsonl"


# ==========================================
# Part 2: No-Think + Skill Prompt (all benchmarks)
# ==========================================
echo ""
echo "=========================================="
echo " Part 2: No-Think + Skill Prompt"
echo "=========================================="

SKILL="skills/skill_combined.md"
SKILL_OUT="outputs/qwen3_14b_bench_skill_prompt"
mkdir -p "$SKILL_OUT"

declare -A BENCH_FILES
BENCH_FILES["MAMO_ComplexLP"]="data/benchmarks/MAMO_ComplexLP_fixed.jsonl"
BENCH_FILES["MAMO_EasyLP"]="data/benchmarks/MAMO_EasyLP_fixed.jsonl"
BENCH_FILES["IndustryOR"]="data/benchmarks/IndustryOR_fixedV2.jsonl"
BENCH_FILES["OptMATH_166"]="data/benchmarks/OptMATH_Bench_166.jsonl"
BENCH_FILES["NL4OPT"]="data/benchmarks/NL4OPT.jsonl"
BENCH_FILES["OptiBench"]="data/benchmarks/OptiBench.jsonl"
BENCH_FILES["MILP_eval"]="data/benchmarks/MILP_all_eval.jsonl"

for NAME in MAMO_ComplexLP MAMO_EasyLP IndustryOR OptMATH_166 NL4OPT OptiBench MILP_eval; do
  DATA="${BENCH_FILES[$NAME]}"
  N=$(wc -l < "$DATA")
  echo ""
  echo "[$NAME] $N items — $DATA"

  # Skill prompt
  echo "  Running No-Think + Skill prompt..."
  python3 scripts/bench_skill_prompt.py "$DATA" "$SKILL_OUT/${NAME}_skill.jsonl" \
    --skill-prompt "$SKILL" --config "$CONFIG"

  # Baseline (no skill)
  echo "  Running No-Think baseline..."
  python3 scripts/bench_skill_prompt.py "$DATA" "$SKILL_OUT/${NAME}_baseline.jsonl" \
    --config "$CONFIG"
done


# ==========================================
# Final Summary
# ==========================================
echo ""
echo "=========================================="
echo " FINAL SUMMARY: Qwen3-14B All Benchmarks"
echo "=========================================="

python3 -c "
import json, os, sys
sys.path.insert(0, 'src')

benchmarks = [
    ('MAMO_ComplexLP', 'MAMO_ComplexLP_fixed'),
    ('MAMO_EasyLP', 'MAMO_EasyLP_fixed'),
    ('IndustryOR', 'IndustryOR_fixedV2'),
    ('OptMATH_166', 'OptMATH_Bench_166'),
    ('NL4OPT', 'NL4OPT'),
    ('OptiBench', 'OptiBench'),
    ('MILP_eval', 'MILP_all_eval'),
]

print()
print('=== Part 1: Self-Evolution Pipeline ===')
print(f'{\"Benchmark\":<20} {\"Think\":>10} {\"NoThink+Skill\":>15} {\"NoThink\":>10} {\"Skills\":>8} {\"Delta\":>8}')
print('-' * 75)

for name, _ in benchmarks:
    d = f'outputs/qwen3_14b_{name}'
    results = {}
    for label, fname in [('skill', 'predictions_skill.jsonl'), ('base', 'predictions_nothink.jsonl')]:
        path = f'{d}/{fname}'
        if os.path.exists(path):
            rows = [json.loads(l) for l in open(path)]
            c = sum(1 for r in rows if r.get('is_correct') == 1)
            results[label] = (c, len(rows))
        else:
            results[label] = (0, 0)

    # Think accuracy
    trace_path = f'{d}/verified_traces.jsonl'
    trace_total_path = f'{d}/traces.jsonl'
    if os.path.exists(trace_path) and os.path.exists(trace_total_path):
        vt = sum(1 for l in open(trace_path))
        tt = sum(1 for l in open(trace_total_path))
        think_str = f'{vt}/{tt} ({vt/max(tt,1)*100:.1f}%)'
    else:
        think_str = 'N/A'

    # Skill count
    sk_path = f'{d}/skills.jsonl'
    if os.path.exists(sk_path):
        sks = [json.loads(l) for l in open(sk_path)]
        sk_ok = sum(1 for s in sks if s.get('status') == 'success')
        sk_str = str(sk_ok)
    else:
        sk_str = 'N/A'

    sc, sn = results['skill']
    bc, bn = results['base']
    sp = sc/sn*100 if sn else 0
    bp = bc/bn*100 if bn else 0
    delta = sp - bp

    print(f'{name:<20} {think_str:>10} {sc}/{sn} ({sp:.1f}%){\"\":>1} {bc}/{bn} ({bp:.1f}%){\"\":>1} {sk_str:>8} {delta:+.1f}%')

print()
print('=== Part 2: No-Think + Skill Prompt ===')
print(f'{\"Benchmark\":<20} {\"N\":>5} {\"Baseline\":>10} {\"Skill\":>10} {\"Delta\":>8}')
print('-' * 58)

for name, _ in benchmarks:
    d = 'outputs/qwen3_14b_bench_skill_prompt'
    for mode in ['baseline', 'skill']:
        path = f'{d}/{name}_{mode}.jsonl'
        try:
            rows = [json.loads(l) for l in open(path)]
            c = sum(1 for r in rows if r.get('is_correct') == 1)
            n = len(rows)
            if mode == 'baseline':
                b_acc, b_n = c / max(n, 1) * 100, n
            else:
                s_acc = c / max(n, 1) * 100
        except:
            if mode == 'baseline':
                b_acc, b_n = 0, 0
            else:
                s_acc = 0

    try:
        delta = s_acc - b_acc
        sign = '+' if delta >= 0 else ''
        print(f'{name:<20} {b_n:>5} {b_acc:>9.1f}% {s_acc:>9.1f}% {sign}{delta:>6.1f}%')
    except:
        pass

print()
print('DONE')
"
