#!/bin/bash
set -e

export HF_HUB_OFFLINE=1

CONFIG="configs/default.yaml"

run_pipeline() {
  local BENCH_NAME=$1
  local DATA_FILE=$2
  local OUT_DIR="outputs/oracle_${BENCH_NAME}"

  mkdir -p "$OUT_DIR"

  local DATA_N=$(wc -l < "$DATA_FILE")

  echo "=========================================="
  echo "Benchmark: $BENCH_NAME ($DATA_N items)"
  echo "Data: $DATA_FILE"
  echo "Output: $OUT_DIR"
  echo "=========================================="

  # --- Think config ---
  python3 -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['inference']['no_think'] = False
cfg['inference']['max_tokens'] = 16384
with open('$OUT_DIR/config_think.yaml', 'w') as f:
    yaml.dump(cfg, f)
"

  # Step 1: Think inference on ALL data
  if [ ! -f "$OUT_DIR/traces.jsonl" ]; then
    echo "[Step 1] Think inference on all data..."
    python3 scripts/baseline_no_think.py "$DATA_FILE" "$OUT_DIR/traces.jsonl" --config "$OUT_DIR/config_think.yaml"
  else
    echo "[Step 1] traces.jsonl exists, skipping."
  fi

  # Step 2: Verify
  if [ ! -f "$OUT_DIR/verified_traces.jsonl" ]; then
    echo "[Step 2] Verifying traces..."
    python3 scripts/3_verify_answers.py "$OUT_DIR/traces.jsonl" "$OUT_DIR/verified_traces.jsonl" --rejected-file "$OUT_DIR/rejected_traces.jsonl"
  else
    echo "[Step 2] verified_traces.jsonl exists, skipping."
  fi

  # Step 3: Distill skills
  if [ ! -f "$OUT_DIR/skills.jsonl" ]; then
    echo "[Step 3] Distilling skills..."
    python3 scripts/4_distill_skills.py "$OUT_DIR/verified_traces.jsonl" prompts/skill_distill.txt "$OUT_DIR/skills.jsonl" --config "$CONFIG"
  else
    echo "[Step 3] skills.jsonl exists, skipping."
  fi

  # Step 4: Skill + NoThink on ALL data
  if [ ! -f "$OUT_DIR/predictions_skill.jsonl" ]; then
    echo "[Step 4] Skill + NoThink inference..."
    python3 scripts/5_retrieve_and_infer.py "$DATA_FILE" "$OUT_DIR/skills.jsonl" prompts/skill_infer.txt "$OUT_DIR/predictions_skill.jsonl" --config "$CONFIG"
  else
    echo "[Step 4] predictions_skill.jsonl exists, skipping."
  fi

  # Step 5: NoThink baseline on ALL data
  if [ ! -f "$OUT_DIR/predictions_nothink.jsonl" ]; then
    echo "[Step 5] NoThink baseline..."
    python3 scripts/baseline_no_think.py "$DATA_FILE" "$OUT_DIR/predictions_nothink.jsonl" --config "$CONFIG"
  else
    echo "[Step 5] predictions_nothink.jsonl exists, skipping."
  fi

  # Report
  echo ""
  echo "=== ${BENCH_NAME} Results ==="
  python3 -c "
import json, sys
sys.path.insert(0, 'src')
from data import answers_match
vt = [json.loads(l) for l in open('$OUT_DIR/verified_traces.jsonl')]
sk = [json.loads(l) for l in open('$OUT_DIR/skills.jsonl')]
sk_ok = sum(1 for s in sk if s.get('status') == 'success')
print(f'  Data: $DATA_N | Verified: {len(vt)} | Skills: {sk_ok}')
for label, fname in [('Think16k(trace)', 'traces.jsonl'), ('Skill+NoThink', 'predictions_skill.jsonl'), ('NoThink', 'predictions_nothink.jsonl')]:
    import os
    fpath = '$OUT_DIR/' + fname
    if not os.path.exists(fpath): continue
    rows = [json.loads(l) for l in open(fpath)]
    c = sum(1 for r in rows if answers_match(r.get('predicted_answer',''), r.get('answer',''))  )
    print(f'  {label:20s}: {c}/{len(rows)} = {c/len(rows)*100:.1f}%')
"
  echo ""
}

# ============================================================
# All benchmarks, full data
# ============================================================

run_pipeline "MAMO_EasyLP"    "data/benchmarks/MAMO_EasyLP_fixed.jsonl"
run_pipeline "MAMO_ComplexLP"  "data/benchmarks/MAMO_ComplexLP_fixed.jsonl"
run_pipeline "IndustryOR"      "data/benchmarks/IndustryOR_fixedV2.jsonl"
run_pipeline "OptMATH"         "data/benchmarks/OptMATH_Bench_166.jsonl"
run_pipeline "MILP"            "data/benchmarks/MILP_all_eval.jsonl"
run_pipeline "NL4OPT"          "data/benchmarks/NL4OPT.jsonl"
run_pipeline "OptiBench"       "data/benchmarks/OptiBench.jsonl"

echo ""
echo "=========================================="
echo "FINAL SUMMARY"
echo "=========================================="
python3 -c "
import json, sys, os
sys.path.insert(0, 'src')
from data import answers_match

benchmarks = ['MAMO_EasyLP', 'MAMO_ComplexLP', 'IndustryOR', 'OptMATH', 'MILP', 'NL4OPT', 'OptiBench']
print()
print(f'{\"Benchmark\":20s} | {\"N\":>5s} | {\"Verified\":>8s} | {\"Skills\":>6s} | {\"Think16k\":>10s} | {\"Skill+NT\":>10s} | {\"NoThink\":>10s} | {\"Delta\":>8s}')
print('-' * 100)

for name in benchmarks:
    d = f'outputs/oracle_{name}'
    if not os.path.exists(d): continue
    vt = [json.loads(l) for l in open(f'{d}/verified_traces.jsonl')]
    sk = [json.loads(l) for l in open(f'{d}/skills.jsonl')]
    sk_ok = sum(1 for s in sk if s.get('status') == 'success')

    results = {}
    for label, fname in [('think', 'traces.jsonl'), ('skill', 'predictions_skill.jsonl'), ('nothink', 'predictions_nothink.jsonl')]:
        fpath = f'{d}/{fname}'
        if not os.path.exists(fpath): continue
        rows = [json.loads(l) for l in open(fpath)]
        c = sum(1 for r in rows if answers_match(r.get('predicted_answer',''), r.get('answer','')))
        results[label] = (c, len(rows))

    def fmt(key):
        if key not in results: return '---'
        c, n = results[key]
        return f'{c/n*100:.1f}%'

    delta = ''
    if 'skill' in results and 'nothink' in results:
        cs, ns = results['skill']
        cn, nn = results['nothink']
        d_pp = cs/ns*100 - cn/nn*100
        delta = f'{d_pp:+.1f}pp'

    n_data = results.get('think', results.get('skill', (0,0)))[1]
    print(f'{name:20s} | {n_data:5d} | {len(vt):8d} | {sk_ok:6d} | {fmt(\"think\"):>10s} | {fmt(\"skill\"):>10s} | {fmt(\"nothink\"):>10s} | {delta:>8s}')
"
