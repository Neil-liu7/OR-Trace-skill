# OR-Trace-Skill Evaluation Pipeline

## Overview

This project evaluates LLMs on 7 Operations Research benchmarks using code generation + execution. The pipeline supports three evaluation modes:
- **NoThink+Code**: Direct code generation without thinking
- **Think+Code**: Code generation with extended thinking (reasoning)
- **NoThink+Skill+Code**: BM25-retrieved skills augment the prompt, then direct code generation

## Quick Start

### 1. Start vLLM Servers

```bash
# 3-server setup on 6 GPUs (TP=2 each)
CUDA_VISIBLE_DEVICES=0,1 python3 -m vllm.entrypoints.openai.api_server \
  --model model/Qwen3-14B --port 8001 --tensor-parallel-size 2 \
  --max-model-len 32768 --trust-remote-code \
  --enable-reasoning --reasoning-parser deepseek_r1 \
  --gpu-memory-utilization 0.9 &

CUDA_VISIBLE_DEVICES=3,4 python3 -m vllm.entrypoints.openai.api_server \
  --model model/Qwen3-14B --port 8002 --tensor-parallel-size 2 \
  --max-model-len 32768 --trust-remote-code \
  --enable-reasoning --reasoning-parser deepseek_r1 \
  --gpu-memory-utilization 0.9 &

CUDA_VISIBLE_DEVICES=5,6 python3 -m vllm.entrypoints.openai.api_server \
  --model model/Qwen3-14B --port 8003 --tensor-parallel-size 2 \
  --max-model-len 32768 --trust-remote-code \
  --enable-reasoning --reasoning-parser deepseek_r1 \
  --gpu-memory-utilization 0.9 &
```

Verify: `curl -s http://127.0.0.1:8001/health`

### 2. Run Evaluations

```bash
# NoThink + Code (4 trials, temp=0.7)
bash scripts/run_nothink_code_pipeline_14b.sh

# NoThink + Skill + Code (4 trials, temp=0.7)
bash scripts/run_nothink_skill_code_pipeline_14b.sh

# NoThink + Skill + Code Greedy (1 trial, temp=0)
bash scripts/run_nothink_skill_code_greedy_14b.sh

# Think + Code (4 trials, temp=0.7)
bash scripts/run_think_code_pipeline_14b_v2.sh
```

### 3. Check Results

```bash
# Count completed trials
for bench in optibench industryor easylp optmath nlp4lp nl4opt complexlp; do
  count=$(ls outputs/<OUT_DIR>/${bench}/trial*.jsonl 2>/dev/null | wc -l)
  echo "$bench: $count/4"
done
```

---

## Project Structure

```
OR-Trace-Skill/
├── scripts/
│   ├── eval_think_code.py          # Think/NoThink + Code evaluation
│   ├── eval_nothink_skill_code.py  # NoThink + Skill + Code evaluation
│   ├── run_nothink_code_pipeline_14b.sh
│   ├── run_nothink_skill_code_pipeline_14b.sh
│   ├── run_nothink_skill_code_greedy_14b.sh
│   └── run_think_code_pipeline_14b_v2.sh
├── configs/
│   ├── nothink_code_port{1,2,3}.yaml      # temp=0.7, no_think
│   ├── nothink_code_greedy_port{1,2,3}.yaml # temp=0.0, no_think
│   └── think_code_port{1,2,3}.yaml         # temp=0.7, think
├── prompts/
│   └── nothink_code.txt            # Code generation prompt template
├── src/
│   ├── api.py                      # Async LLM API client
│   ├── data.py                     # Data I/O, prompt rendering, answer matching
│   └── retrieval.py                # BM25 skill retrieval (SkillBank)
├── data/benchmarks/                # 7 OR benchmark JSONL files
└── outputs/                        # Evaluation results
```

---

## Benchmarks

| Dataset | Size | Domain | File |
|---------|------|--------|------|
| OptiBench | 605 | General Optimization | `data/benchmarks/OptiBench.jsonl` |
| IndustryOR | 100 | Industrial OR | `data/benchmarks/IndustryOR_fixedV2.jsonl` |
| MAMO_EasyLP | 642 | Easy Linear Programming | `data/benchmarks/MAMO_EasyLP_fixed.jsonl` |
| MAMO_ComplexLP | 203 | Complex LP | `data/benchmarks/MAMO_ComplexLP_fixed.jsonl` |
| OptMATH | 166 | Mixed OR/Math | `data/benchmarks/OptMATH_Bench_166.jsonl` |
| NL4OPT | 245 | Natural Language → Optimization | `data/benchmarks/NL4OPT_Test.jsonl` |
| NLP4LP | 322 | NLP for LP | `data/benchmarks/NLP4LP_full.jsonl` |

**Data format** (each line in JSONL):
```json
{"question_id": "MAMO_EasyLP_fixed_0000", "benchmark": "MAMO_EasyLP_fixed", "question": "A marketing company...", "answer": "10000"}
```

---

## Config YAML Format

```yaml
model: "model/Qwen3-14B"
api_base_url: "http://127.0.0.1:8001/v1/chat/completions"
api_key: "dummy"
no_think: true          # Set to true for NoThink mode; omit for Think mode

trace_generation:
  temperature: 0.7      # 0.0 for greedy, 0.7 for sampling
  max_tokens: 4096      # NoThink needs 4096; Think needs 8192+
  max_concurrent: 16    # NoThink can handle 16; Think use 8
  max_retries: 0
  timeout: 120          # NoThink: 120s; Think: 300s
```

**Port assignment convention**:
- Port 8001: OptiBench, IndustryOR
- Port 8002: EasyLP, OptMATH
- Port 8003: NLP4LP, NL4OPT, ComplexLP

---

## Eval Scripts

### `scripts/eval_think_code.py`

Evaluates Think+Code or NoThink+Code (controlled by config `no_think` flag).

```bash
python3 scripts/eval_think_code.py <input.jsonl> <prompt.txt> <output.jsonl> --config <config.yaml> [--limit N]
```

### `scripts/eval_nothink_skill_code.py`

Evaluates NoThink+Skill+Code with BM25 skill retrieval.

```bash
python3 scripts/eval_nothink_skill_code.py <input.jsonl> <prompt.txt> <output.jsonl> \
  --skills <skills_filtered.jsonl> --config <config.yaml> --top-k 3 [--limit N]
```

**Answer extraction priority** (4-level fallback):
1. Code block in response → execute → parse stdout
2. Code block in reasoning → execute → parse stdout
3. Text extraction from response (regex patterns)
4. Text extraction from reasoning

**Answer matching**: `answers_match(predicted, gold, rel_tol=0.01)` — numeric comparison with 1% relative tolerance.

---

## Skill Files

**Location**: `outputs/qwen3_14b_code_{BenchmarkName}/skills_filtered.jsonl`

| Benchmark | Skill File | Count |
|-----------|-----------|-------|
| OptiBench | `outputs/qwen3_14b_code_OptiBench/skills_filtered.jsonl` | 1 |
| IndustryOR | `outputs/qwen3_14b_code_IndustryOR/skills_filtered.jsonl` | 5 |
| EasyLP | `outputs/qwen3_14b_code_MAMO_EasyLP/skills_filtered.jsonl` | 6 |
| ComplexLP | `outputs/qwen3_14b_code_MAMO_ComplexLP/skills_filtered.jsonl` | 5 |
| OptMATH | `outputs/qwen3_14b_code_OptMATH/skills_filtered.jsonl` | 1 |
| NLP4LP | `outputs/qwen3_14b_code_NLP4LP/skills_filtered.jsonl` | 1 |
| NL4OPT | `outputs/qwen3_14b_code_NL4OPT/skills_filtered.jsonl` | 4 |

**Key fields in skill records**:
- `inject_text`: Combined modeling pattern + code template (used as `{SOLVING_HINTS}`)
- `procedure`: Step-by-step solving procedure
- `code_template`: Executable code snippet
- `keywords`: BM25-friendly retrieval terms

---

## Prompt Template

`prompts/nothink_code.txt`:
```
You are an expert OR solver. Write a complete, executable Python script...

[Modeling Patterns and Code Templates]
{SOLVING_HINTS}              ← filled by retrieved skills (or empty if no skill)
[/Modeling Patterns and Code Templates]

Problem:
{PROBLEM}                    ← filled by test question
```

The `render_prompt()` function in `src/data.py` handles substitution.

---

## Output Format

Each trial produces a JSONL file with one record per question:

```json
{
  "question_id": "...",
  "question": "...",
  "answer": "10000",
  "gen_model": "model/Qwen3-14B",
  "mode": "nothink_skill_code",
  "n_skills_retrieved": 3,
  "raw_reasoning": "",
  "raw_model_response": "```python\nfrom pulp import...\n```",
  "predicted_answer": "10000",
  "answer_source": "code_response",
  "is_correct": 1,
  "prompt_tokens": 1234,
  "completion_tokens": 456,
  "total_tokens": 1690,
  "status": "success",
  "timestamp": "2026-05-10T..."
}
```

**`answer_source` values**: `code_response`, `code_reasoning`, `text_response`, `text_reasoning`, `none`

---

## Results Aggregation

### pass@k Computation

```python
from math import comb

def pass_at_k(n, c, k):
    """n=total trials, c=correct count for a question, k=target"""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)
```

### Aggregate Script Pattern

```python
import json
benchmarks = [('EasyLP', 'easylp'), ('IndustryOR', 'industryor'), ...]
base = 'outputs/<dir>'
n_trials = 4

for name, dir_name in benchmarks:
    trials = []
    for i in range(1, n_trials + 1):
        rows = [json.loads(l) for l in open(f'{base}/{dir_name}/trial{i}.jsonl')]
        trials.append(rows)
    # Compute per-question correct counts, then pass@k
```

---

## Adding a New Model

1. **Serve the model** via vLLM (or any OpenAI-compatible API)
2. **Create config YAML** — change `model` field and optionally adjust `max_tokens`/`timeout`
3. **Run pipeline** — same scripts work unchanged
4. **Handle thinking mode** — if model doesn't support thinking, set `no_think: true` or remove `--enable-reasoning` from vLLM

Example for a new model:
```yaml
# configs/new_model_port1.yaml
model: "model/NewModel-7B"
api_base_url: "http://127.0.0.1:8001/v1/chat/completions"
api_key: "dummy"
no_think: true

trace_generation:
  temperature: 0.7
  max_tokens: 4096
  max_concurrent: 16
  max_retries: 0
  timeout: 120
```

---

## Skill Generation Pipeline (Full)

For generating skills from scratch for a new benchmark:

```
Training Problems → [Think Traces] → [Verify Correct] → [Distill Skills] → [LOO Eval] → [Filter & Merge] → skills_filtered.jsonl
```

Key scripts:
- `scripts/2_generate_traces.py` — Generate thinking traces
- `scripts/3_verify_answers.py` — Keep only correct traces
- `scripts/4_distill_skills.py` — Extract skills from traces
- `scripts/5_retrieve_and_infer.py` — Leave-one-out evaluation
- `scripts/filter_and_merge_skills.py` — Filter harmful skills, merge similar ones

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| Pipeline hangs | vLLM OOM | Reduce `max_concurrent` or `max_model_len` |
| 0% code exec rate | Wrong prompt format | Check `{SOLVING_HINTS}` substitution |
| Low accuracy on OptMATH | Problems need math reasoning, not just LP | Expected behavior |
| Port contention | Multiple pipelines sharing ports | Run one pipeline at a time per port |
| Greedy < Sampling | Complex problems need diversity | Use sampling (temp=0.7) + pass@k |

---

## Performance Reference (Qwen3-14B)

| Mode | Tokens/Question | Concurrency | Speed |
|------|----------------|-------------|-------|
| NoThink | ~430 | 16 | ~seconds/question |
| Think | ~4650 | 8 | ~37-89s/question |
| NoThink+Skill | ~430 | 16 | ~seconds/question |
