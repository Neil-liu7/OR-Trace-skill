#!/bin/bash
set -e

export HF_HUB_OFFLINE=1

DATA_FILE="data/benchmarks/MAMO_ComplexLP_fixed.jsonl"
SKILL_FILE="outputs/oracle_MAMO_ComplexLP/skills.jsonl"
OUT_DIR="outputs/oracle_MAMO_ComplexLP"

N_RUNS=5

# Create config with temperature=0.7
python3 -c "
import yaml
with open('configs/default.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['inference']['temperature'] = 0.7
cfg['inference']['no_think'] = True
cfg['inference']['max_tokens'] = 8192
with open('$OUT_DIR/config_t07.yaml', 'w') as f:
    yaml.dump(cfg, f)
"

CONFIG="$OUT_DIR/config_t07.yaml"

echo "=========================================="
echo "MAMO_ComplexLP 5-trial (temperature=0.7)"
echo "Skills: $SKILL_FILE"
echo "=========================================="

for i in $(seq 1 $N_RUNS); do
  echo ""
  echo "--- Run $i / $N_RUNS ---"

  if [ ! -f "$OUT_DIR/predictions_skill_run${i}.jsonl" ]; then
    echo "  [Skill+NoThink] run $i ..."
    python3 scripts/5_retrieve_and_infer.py "$DATA_FILE" "$SKILL_FILE" prompts/skill_infer.txt "$OUT_DIR/predictions_skill_run${i}.jsonl" --config "$CONFIG" 2>&1
  else
    echo "  [Skill+NoThink] run $i exists, skipping."
  fi

  if [ ! -f "$OUT_DIR/predictions_nothink_run${i}.jsonl" ]; then
    echo "  [NoThink] run $i ..."
    python3 scripts/baseline_no_think.py "$DATA_FILE" "$OUT_DIR/predictions_nothink_run${i}.jsonl" --config "$CONFIG" 2>&1
  else
    echo "  [NoThink] run $i exists, skipping."
  fi
done

echo ""
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
python3 -c "
import json, sys, os
sys.path.insert(0, 'src')
from data import answers_match

out = '$OUT_DIR'
n_runs = $N_RUNS

skill_accs = []
nothink_accs = []

for i in range(1, n_runs + 1):
    for label, prefix, acc_list in [('Skill+NT', 'predictions_skill_run', skill_accs), ('NoThink', 'predictions_nothink_run', nothink_accs)]:
        fpath = f'{out}/{prefix}{i}.jsonl'
        if not os.path.exists(fpath):
            continue
        rows = [json.loads(l) for l in open(fpath)]
        c = sum(1 for r in rows if answers_match(r.get('predicted_answer',''), r.get('answer','')))
        acc = c / len(rows) * 100
        acc_list.append((i, c, len(rows), acc))
        print(f'  {label} run{i}: {c}/{len(rows)} = {acc:.1f}%')

print()
if skill_accs:
    avg_s = sum(a for _,_,_,a in skill_accs) / len(skill_accs)
    std_s = (sum((a - avg_s)**2 for _,_,_,a in skill_accs) / len(skill_accs))**0.5
    print(f'Skill+NoThink: {avg_s:.1f}% +/- {std_s:.1f}%')
if nothink_accs:
    avg_n = sum(a for _,_,_,a in nothink_accs) / len(nothink_accs)
    std_n = (sum((a - avg_n)**2 for _,_,_,a in nothink_accs) / len(nothink_accs))**0.5
    print(f'NoThink:       {avg_n:.1f}% +/- {std_n:.1f}%')
if skill_accs and nothink_accs:
    print(f'Delta:         {avg_s - avg_n:+.1f}pp')
"
