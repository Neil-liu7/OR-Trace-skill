#!/bin/bash
set -e
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

CONFIG="configs/default.yaml"
SKILL="skills/skill_combined.md"
OUT_BASE="outputs/bench_skill_prompt"

mkdir -p "$OUT_BASE"

declare -A BENCHMARKS
BENCHMARKS["MAMO_ComplexLP"]="data/benchmarks/MAMO_ComplexLP_fixed.jsonl"
BENCHMARKS["IndustryOR"]="data/benchmarks/IndustryOR_fixedV2.jsonl"
BENCHMARKS["NL4OPT"]="data/benchmarks/NL4OPT.jsonl"
BENCHMARKS["OptiBench"]="data/benchmarks/OptiBench.jsonl"
BENCHMARKS["OptMATH_166"]="data/benchmarks/OptMATH_Bench_166.jsonl"
BENCHMARKS["MILP_eval"]="data/benchmarks/MILP_all_eval.jsonl"

echo "=========================================="
echo " Skill Prompt vs No-Think Baseline"
echo "=========================================="
echo ""

for NAME in MAMO_ComplexLP IndustryOR NL4OPT OptiBench OptMATH_166 MILP_eval; do
  DATA="${BENCHMARKS[$NAME]}"
  if [ ! -f "$DATA" ]; then
    echo "[$NAME] SKIP — $DATA not found"
    continue
  fi
  N=$(wc -l < "$DATA")
  echo "[$NAME] $N items — $DATA"

  # Skill prompt
  echo "  Running skill prompt..."
  python3 scripts/bench_skill_prompt.py "$DATA" "$OUT_BASE/${NAME}_skill.jsonl" \
    --skill-prompt "$SKILL" --config "$CONFIG"

  # Baseline (no skill)
  echo "  Running baseline..."
  python3 scripts/bench_skill_prompt.py "$DATA" "$OUT_BASE/${NAME}_baseline.jsonl" \
    --config "$CONFIG"

  echo ""
done

# Summary table
echo "=========================================="
echo " SUMMARY"
echo "=========================================="
python3 -c "
import json, sys
sys.path.insert(0, 'src')
from data import answers_match, extract_final_answer

benchmarks = [
    ('MAMO_ComplexLP', 203),
    ('IndustryOR', 100),
    ('NL4OPT', 245),
    ('OptiBench', 605),
    ('OptMATH_166', 166),
    ('MILP_eval', 360),
]

print(f'{\"Benchmark\":<20} {\"N\":>5} {\"Baseline\":>10} {\"Skill\":>10} {\"Delta\":>8}')
print('-' * 58)

for name, _ in benchmarks:
    for mode in ['baseline', 'skill']:
        path = f'$OUT_BASE/{name}_{mode}.jsonl'
        try:
            rows = [json.loads(l) for l in open(path)]
        except FileNotFoundError:
            continue
        c = sum(1 for r in rows if r.get('is_correct') == 1)
        n = len(rows)
        if mode == 'baseline':
            b_acc = c / max(n, 1) * 100
            b_n = n
        else:
            s_acc = c / max(n, 1) * 100

    try:
        delta = s_acc - b_acc
        sign = '+' if delta >= 0 else ''
        print(f'{name:<20} {b_n:>5} {b_acc:>9.1f}% {s_acc:>9.1f}% {sign}{delta:>6.1f}%')
    except:
        pass
"
echo ""
echo "DONE"
