#!/bin/bash
set -e

export HF_HUB_OFFLINE=1

BENCH="MAMO_ComplexLP_fixed"
DATA="data/benchmarks/MAMO_ComplexLP_fixed.jsonl"
OUT_DIR="outputs/${BENCH}_full"
SKILL_FILE="$OUT_DIR/skills.jsonl"
CONFIG="configs/default.yaml"

# Generate config with temperature=0.1
python3 -c "
import yaml
with open('$CONFIG') as f:
    cfg = yaml.safe_load(f)
cfg['inference']['temperature'] = 0.1
cfg['inference']['no_think'] = True
cfg['inference']['max_tokens'] = 8192
with open('$OUT_DIR/config_t01.yaml', 'w') as f:
    yaml.dump(cfg, f)
"

echo "=========================================="
echo "5-run experiment: ComplexLP, temperature=0.1"
echo "=========================================="

for i in 1 2 3 4 5; do
  echo ""
  echo "--- Run $i/5: Skill (No-Think) ---"
  python3 scripts/5_retrieve_and_infer.py \
    "$DATA" "$SKILL_FILE" prompts/skill_infer.txt \
    "$OUT_DIR/predictions_skill_run${i}.jsonl" \
    --config "$OUT_DIR/config_t01.yaml"

  echo "--- Run $i/5: No-Think baseline ---"
  python3 scripts/baseline_no_think.py \
    "$DATA" "$OUT_DIR/predictions_nothink_run${i}.jsonl" \
    --config "$OUT_DIR/config_t01.yaml"
done

echo ""
echo "=========================================="
echo "Summary (tolerance matching)"
echo "=========================================="
python3 -c "
import json, sys
sys.path.insert(0, 'src')
from data import answers_match

for mode in ['skill', 'nothink']:
    accs = []
    for i in range(1, 6):
        path = '$OUT_DIR/predictions_{}_run{}.jsonl'.format(mode, i)
        rows = [json.loads(l) for l in open(path)]
        c = sum(1 for r in rows if answers_match(r.get('predicted_answer',''), r.get('answer','')))
        acc = c / len(rows) * 100
        accs.append(acc)
        print(f'  {mode} run{i}: {c}/{len(rows)} = {acc:.1f}%')
    avg = sum(accs) / len(accs)
    mn, mx = min(accs), max(accs)
    print(f'  {mode} AVG: {avg:.1f}% (range: {mn:.1f}%-{mx:.1f}%)')
    print()
"
