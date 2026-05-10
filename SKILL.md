# OR-Trace-Skill Evaluation Pipeline — Operator Guide

> This document is designed for AI assistants (Claude, etc.) to read and follow when running evaluations on new models. It provides step-by-step instructions for the complete evaluation workflow.

---

## Task: Evaluate a New Model

When the user says "帮我测评 X 模型", follow this workflow:

### Step 1: Determine Model Setup

Ask the user:
1. Model path (e.g., `model/Qwen3-14B`, `model/DeepSeek-R1-7B`)
2. How many GPUs available and which ports to use
3. Whether the model supports thinking mode (`--enable-reasoning`)
4. Tensor parallel size needed (14B→TP=2, 7B→TP=1, 72B→TP=4)

### Step 2: Start vLLM Servers

Template command (adapt per model):

```bash
CUDA_VISIBLE_DEVICES=<gpus> python3 -m vllm.entrypoints.openai.api_server \
  --model <model_path> \
  --port <port> \
  --tensor-parallel-size <tp_size> \
  --max-model-len 32768 \
  --trust-remote-code \
  --enable-reasoning --reasoning-parser deepseek_r1 \
  --gpu-memory-utilization 0.9
```

Notes:
- `--enable-reasoning --reasoning-parser deepseek_r1`: Required for Qwen3 thinking mode. Remove for models without thinking support.
- For models using `<think>` tags natively (DeepSeek-R1), the parser extracts thinking automatically.
- For models without thinking, omit these flags and always use `no_think: true` in config.

Verify server: `curl -s http://127.0.0.1:<port>/health`

### Step 3: Create Config Files

Create one config per port. Template:

```yaml
# configs/<model_name>_<mode>_port<N>.yaml
model: "<model_path>"                              # Must match vLLM --model
api_base_url: "http://127.0.0.1:<port>/v1/chat/completions"
api_key: "dummy"
no_think: true                                     # Remove this line for think mode

trace_generation:
  temperature: 0.7    # 0.0 for greedy, 0.7 for sampling/pass@k
  max_tokens: 4096    # NoThink: 4096; Think: 8192-16384
  max_concurrent: 16  # NoThink: 16; Think: 8 (think uses more memory)
  max_retries: 0
  timeout: 120        # NoThink: 120; Think: 300
```

**Config naming convention**: `<model>_<mode>_port<N>.yaml`
- Example: `qwen3_8b_nothink_port1.yaml`, `deepseek_r1_think_port1.yaml`

### Step 4: Create Pipeline Script

Copy and modify an existing pipeline script. The key variables to change:

```bash
OUT_BASE="outputs/<descriptive_name>"   # e.g., "outputs/qwen3_8b_nothink_code"
N_TRIALS=4                               # 4 for pass@k, 1 for greedy
PROMPT="prompts/nothink_code.txt"        # Same prompt for all models
CONFIG_PREFIX="configs/<your_config>"     # Config file pattern
```

**For NoThink+Code** (no skill), use `scripts/eval_think_code.py`:
```bash
python3 scripts/eval_think_code.py \
  "$input" "$PROMPT" "$output" --config "$config"
```

**For NoThink+Skill+Code** (with skill), use `scripts/eval_nothink_skill_code.py`:
```bash
python3 scripts/eval_nothink_skill_code.py \
  "$input" "$PROMPT" "$output" \
  --skills "$skills" --config "$config" --top-k 3
```

**For Think+Code**, use `scripts/eval_think_code.py` with think config (no `no_think` flag):
```bash
python3 scripts/eval_think_code.py \
  "$input" "$PROMPT" "$output" --config "$config"
```

### Step 5: Run and Monitor

```bash
# Launch
nohup bash scripts/<your_pipeline>.sh > outputs/<log_name>.log 2>&1 &

# Monitor progress
for bench in optibench industryor easylp optmath nlp4lp nl4opt complexlp; do
  count=$(ls outputs/<OUT_BASE>/${bench}/trial*.jsonl 2>/dev/null | wc -l)
  echo "$bench: $count/<N_TRIALS>"
done

# Check for running processes
ps aux | grep "eval_think_code\|eval_nothink_skill_code" | grep -v grep
```

### Step 6: Aggregate Results

```python
python3 -c "
import json, sys
from math import comb
sys.path.insert(0, 'src')

def pass_at_k(n, c, k):
    if n - c < k: return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

benchmarks = [
    ('EasyLP', 'easylp'), ('IndustryOR', 'industryor'),
    ('ComplexLP', 'complexlp'), ('NL4OPT', 'nl4opt'),
    ('NLP4LP', 'nlp4lp'), ('OptiBench', 'optibench'),
    ('OptMATH', 'optmath'),
]
base = 'outputs/<OUT_BASE>'
n_trials = 4

for name, dir_name in benchmarks:
    trials = []
    for i in range(1, n_trials + 1):
        try:
            rows = [json.loads(l) for l in open(f'{base}/{dir_name}/trial{i}.jsonl')]
            trials.append(rows)
        except: break
    if not trials: continue
    n = len(trials)
    n_q = len(trials[0])
    per_q = []
    for qi in range(n_q):
        c = sum(1 for ti in range(n) if trials[ti][qi].get('is_correct') == 1)
        per_q.append(c)
    avg = sum(per_q) / (n_q * n) * 100
    p4 = sum(pass_at_k(n, c, 4) for c in per_q) / len(per_q) * 100
    print(f'{name:<12} pass@1={avg:.1f}%  pass@4={p4:.1f}%')
"
```

---

## Benchmarks (7 total)

| Key | Dataset | Size | File Path |
|-----|---------|------|-----------|
| optibench | OptiBench | 605 | `data/benchmarks/OptiBench.jsonl` |
| industryor | IndustryOR | 100 | `data/benchmarks/IndustryOR_fixedV2.jsonl` |
| easylp | MAMO_EasyLP | 642 | `data/benchmarks/MAMO_EasyLP_fixed.jsonl` |
| complexlp | MAMO_ComplexLP | 203 | `data/benchmarks/MAMO_ComplexLP_fixed.jsonl` |
| optmath | OptMATH | 166 | `data/benchmarks/OptMATH_Bench_166.jsonl` |
| nl4opt | NL4OPT | 245 | `data/benchmarks/NL4OPT_Test.jsonl` |
| nlp4lp | NLP4LP | 322 | `data/benchmarks/NLP4LP_full.jsonl` |

Data format per line:
```json
{"question_id": "...", "benchmark": "...", "question": "problem text", "answer": "numeric_answer"}
```

---

## Port Assignment Convention

Load-balance benchmarks across ports by size:
- **Port 1** (8001): OptiBench (605), IndustryOR (100) → 705 questions
- **Port 2** (8002): EasyLP (642), OptMATH (166) → 808 questions
- **Port 3** (8003): NLP4LP (322), NL4OPT (245), ComplexLP (203) → 770 questions

This balances load roughly evenly across 3 servers.

---

## Skill Files (for NoThink+Skill+Code mode)

These are pre-generated and shared across all model evaluations:

| Benchmark | Skill File | Skills |
|-----------|-----------|--------|
| optibench | `outputs/qwen3_14b_code_OptiBench/skills_filtered.jsonl` | 1 |
| industryor | `outputs/qwen3_14b_code_IndustryOR/skills_filtered.jsonl` | 5 |
| easylp | `outputs/qwen3_14b_code_MAMO_EasyLP/skills_filtered.jsonl` | 6 |
| complexlp | `outputs/qwen3_14b_code_MAMO_ComplexLP/skills_filtered.jsonl` | 5 |
| optmath | `outputs/qwen3_14b_code_OptMATH/skills_filtered.jsonl` | 1 |
| nlp4lp | `outputs/qwen3_14b_code_NLP4LP/skills_filtered.jsonl` | 1 |
| nl4opt | `outputs/qwen3_14b_code_NL4OPT/skills_filtered.jsonl` | 4 |

Skills contain `inject_text` field (modeling patterns + code templates) used as `{SOLVING_HINTS}` in the prompt.

---

## Prompt Template

File: `prompts/nothink_code.txt`

```
You are an expert OR solver. Write a complete, executable Python script to solve the problem below.

Choose the appropriate solver:
- PuLP for linear/integer programming (LP/MIP)
- scipy.optimize for nonlinear, quadratic, geometric, or calculus-based optimization

Below are modeling patterns and code templates from similar problems. Use them as reference for how to structure your solution and which solver to use.

[Modeling Patterns and Code Templates]
{SOLVING_HINTS}
[/Modeling Patterns and Code Templates]

Now solve the following problem. Output ONLY a Python code block.
Your code must print the answer in exactly this format:
print(f"Answer: {value}")

Problem:
{PROBLEM}
```

- `{PROBLEM}` → test question text
- `{SOLVING_HINTS}` → retrieved skills joined by `\n\n---\n\n` (empty string if no skill mode)

---

## Eval Script Details

### `scripts/eval_think_code.py`

```
Usage: python3 scripts/eval_think_code.py <input_file> <prompt_file> <output_file> [--config CONFIG] [--limit N]
```

- Reads config `no_think` flag to control thinking mode
- If `no_think: true` → sets `enable_thinking=False` in API call
- If `no_think` absent → uses default (thinking enabled)
- Extracts code from response, executes with 30s timeout
- Falls back to text answer extraction if code fails

### `scripts/eval_nothink_skill_code.py`

```
Usage: python3 scripts/eval_nothink_skill_code.py <input_file> <prompt_file> <output_file> --skills <file> [--config CONFIG] [--top-k K] [--limit N]
```

- Always uses `enable_thinking=False`
- Loads skills into `SkillBank` (BM25 index)
- For each question: retrieves top-k skills → fills `{SOLVING_HINTS}` → generates code
- Same code execution + answer extraction as above

### Answer Extraction Priority

1. Extract ```python block from response → execute → parse `Answer: <value>` from stdout
2. Extract ```python block from reasoning → execute → parse stdout
3. Regex extract from response text (patterns: `\boxed{}`, `**Answer:**`, etc.)
4. Regex extract from reasoning text
5. Return empty (marked as "none")

### Answer Matching

`src/data.py:answers_match(predicted, gold, rel_tol=0.01)`:
- Normalizes both strings (remove $, commas, units)
- Converts to float if possible
- Returns True if `|pred - gold| / |gold| <= 0.01` (1% tolerance)
- Falls back to exact string match for non-numeric

---

## Source Modules

### `src/api.py` — LLM API Client

- `call_llm(session, prompt, *, model, temperature, max_tokens, enable_thinking, timeout, api_base_url, api_key, max_retries)` → `LLMResponse`
- Sends to OpenAI-compatible endpoint (vLLM)
- `enable_thinking` controls `chat_template_kwargs: {"enable_thinking": bool}`
- Extracts reasoning from `reasoning_content` field in response
- Returns: `LLMResponse(status, text, reasoning, prompt_tokens, completion_tokens, total_tokens, error)`

### `src/data.py` — Data Utilities

- `load_jsonl(path)` — Load JSONL (auto gzip support)
- `write_jsonl(path, rows)` — Write JSONL
- `render_prompt(template, *, problem, hints, ...)` — Template substitution
- `extract_final_answer(text)` — Regex-based answer extraction
- `answers_match(candidate, gold, rel_tol=0.01)` — Numeric comparison

### `src/retrieval.py` — Skill Retrieval

- `SkillBank(records)` — Initializes BM25 index from skill records
  - Filters records that have valid `skill_text` (via `preferred_skill_text()`)
  - `preferred_skill_text(row)` returns `inject_text` or `procedure` field
- `SkillBank.search(query, top_k)` → `list[dict]` with `skill_text`, `score`, `question_id`

---

## Complete Example: Evaluate Qwen3-8B

```bash
# 1. Start server (7B model, TP=1, single GPU)
CUDA_VISIBLE_DEVICES=2 python3 -m vllm.entrypoints.openai.api_server \
  --model model/Qwen3-8B --port 8004 \
  --tensor-parallel-size 1 --max-model-len 32768 \
  --trust-remote-code --enable-reasoning --reasoning-parser deepseek_r1 \
  --gpu-memory-utilization 0.9 &

# 2. Create config
cat > configs/qwen3_8b_nothink_port4.yaml << 'EOF'
model: "model/Qwen3-8B"
api_base_url: "http://127.0.0.1:8004/v1/chat/completions"
api_key: "dummy"
no_think: true

trace_generation:
  temperature: 0.7
  max_tokens: 4096
  max_concurrent: 16
  max_retries: 0
  timeout: 120
EOF

# 3. Run single benchmark test
python3 scripts/eval_think_code.py \
  data/benchmarks/IndustryOR_fixedV2.jsonl \
  prompts/nothink_code.txt \
  outputs/qwen3_8b_nothink/industryor/trial1.jsonl \
  --config configs/qwen3_8b_nothink_port4.yaml --limit 5

# 4. Verify output
cat outputs/qwen3_8b_nothink/industryor/trial1.jsonl | python3 -c "
import json, sys
for line in sys.stdin:
    r = json.loads(line)
    print(f\"{r['question_id']}: correct={r['is_correct']}, source={r['answer_source']}\")
"
```

---

## Existing Results (Qwen3-14B Reference)

| Benchmark | NoThink+Code | NoThink+Skill+Code | Think+Code |
|-----------|:---:|:---:|:---:|
| EasyLP | 69.4% | 86.3% | 85.9% |
| IndustryOR | 35.5% | 42.2% | 51.0% |
| ComplexLP | 29.4% | 32.0% | 45.6% |
| NL4OPT | 79.5% | 81.1% | 80.4% |
| NLP4LP | 59.9% | 63.1% | 62.9% |
| OptiBench | 63.0% | 62.9% | 65.8% |
| OptMATH | 11.7% | 13.3% | 13.0% |

Settings: temperature=0.7, 4 trials, pass@1 average.

---

## Key Design Decisions

1. **Same prompt for all modes** — `prompts/nothink_code.txt` is shared; `{SOLVING_HINTS}` is empty when no skill
2. **Code execution as primary answer source** — More reliable than text extraction for numeric answers
3. **BM25 for skill retrieval** — Zero-cost, no GPU needed, works well for structured problems
4. **4 trials at temp=0.7** — Enables pass@k computation; greedy (temp=0) for deterministic baseline
5. **3-port parallelism** — Balances GPU utilization across benchmarks of different sizes
6. **Skills are model-agnostic** — Same skill files can be used across different models
