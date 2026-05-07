# OR-Trace-Skill: Self-Evolution Framework for Operations Research

A self-evolution framework that extracts reusable solving skills from LLM reasoning traces and iteratively refines them through feedback-driven filtering.

## Overview

The framework enables small LLMs to solve complex Operations Research problems more effectively by:
1. Generating reasoning traces on training problems (Think mode)
2. Distilling procedural skills from correct traces
3. Evaluating skill effectiveness via Leave-One-Out (LOO) analysis
4. Filtering harmful/useless skills and merging similar ones
5. Deploying refined skills via retrieval-augmented inference on test problems

## Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TRAINING PHASE                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Training Set ──→ [Think-mode Trace Generation] ──→ Traces          │
│                           │                                         │
│                           ▼                                         │
│                   [Answer Verification]                              │
│                           │                                         │
│                           ▼                                         │
│                 Verified Correct Traces                              │
│                           │                                         │
│                           ▼                                         │
│                   [Skill Distillation]                               │
│                           │                                         │
│                           ▼                                         │
│                      skills_raw.jsonl                                │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                     SELF-EVALUATION PHASE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Training Set ──→ [LOO Inference]          (use other skills)       │
│                        │                                            │
│  Training Set ──→ [Baseline Inference]     (no skills)              │
│                        │                                            │
│                        ▼                                            │
│              Per-skill Effectiveness Metrics                         │
│              (NetScore = HelpRate - HurtRate)                        │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                     FILTER & MERGE PHASE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  skills_raw ──→ [Filter: remove harmful & useless skills]           │
│                        │                                            │
│                        ▼                                            │
│              [Merge: cluster similar skills by embedding]            │
│                        │                                            │
│                        ▼                                            │
│                  skills_filtered.jsonl                               │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                       TEST PHASE                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Test Problem ──→ [Skill Retrieval] ──→ Top-K Similar Skills        │
│                        │                                            │
│                        ▼                                            │
│              [No-Think Inference with Skill Prompt]                  │
│                        │                                            │
│                        ▼                                            │
│                   Prediction                                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Trace Generation (`scripts/2_generate_traces.py`)

Uses Think mode (extended reasoning) to generate step-by-step solutions for training problems. The model produces full reasoning chains that serve as the raw material for skill extraction.

### 2. Answer Verification (`scripts/3_verify_answers.py`)

Compares model predictions against gold answers using tolerance-based matching. Only traces that lead to correct answers are kept for skill distillation.

### 3. Skill Distillation (`scripts/4_distill_skills.py`)

Extracts reusable procedural skills from verified traces. Each skill contains:
- **Solving Procedure**: Generalized step-by-step actions (not strategic advice)
- **Worked Example**: Compact computation template (~150-250 words)
- **Retrieval Keywords**: BM25-friendly terms for matching similar problems

### 4. LOO Evaluation + Filtering (`scripts/filter_and_merge_skills.py`)

The core self-evaluation mechanism:

**Leave-One-Out (LOO) Inference**: For each training problem, retrieve skills from *other* problems (excluding self) and run inference. Compare with a no-skill baseline to measure each skill's impact.

**Per-Skill Metrics**:
- `NetScore = HelpRate - HurtRate`
- A skill *helps* when baseline fails but LOO succeeds
- A skill *hurts* when baseline succeeds but LOO fails

**Filtering Rules**:
- Remove skills with `NetScore < 0` (harmful)
- Remove skills with `HelpRate == 0` and high retrieval count (useless but frequently retrieved)
- Merge skills with cosine similarity > 0.85 (deduplification)

### 5. Skill Retrieval & Inference (`scripts/5_retrieve_and_infer.py`)

For test problems, retrieves the most relevant skills using hybrid search (BM25 + structural pre-filtering + embedding RRF), then injects them into the prompt for no-think inference.

**Retrieval Methods**:
- `bm25`: Classic keyword matching
- `structural`: Problem-type pre-filtering + BM25
- `hybrid`: BM25 + embedding with Reciprocal Rank Fusion
- `structural_hybrid`: Structural pre-filtering + hybrid RRF (default)

## Quick Start

```bash
# 1. Start vLLM server
CUDA_VISIBLE_DEVICES=5,6 python3 -m vllm.entrypoints.openai.api_server \
  --model model/Qwen3-14B --port 8001 --tensor-parallel-size 2 \
  --max-model-len 32768 --trust-remote-code \
  --enable-reasoning --reasoning-parser deepseek_r1

# 2. Run full pipeline
bash scripts/run_optmath_pipeline.sh
```

## Configuration

See `configs/optmath_14b.yaml` for a full example. Key settings:

```yaml
model: "model/Qwen3-14B"
api_base_url: "http://127.0.0.1:8001/v1/chat/completions"

trace_generation:
  temperature: 0.7
  max_tokens: 16384      # Think mode needs more tokens

inference:
  temperature: 0.0
  max_tokens: 16384
  no_think: true          # No-think for skill-guided inference
  top_k: 2               # Retrieve top-2 skills per problem
  retrieval:
    method: structural_hybrid
    embed_model: BAAI/bge-small-en-v1.5
```

## Project Structure

```
OR-Trace-Skill/
├── configs/                    # YAML configurations
├── data/benchmarks/            # Benchmark datasets (JSONL)
├── prompts/
│   ├── source_cot.txt          # Think-mode trace generation prompt
│   ├── skill_distill.txt       # Skill extraction prompt
│   ├── skill_infer.txt         # Skill-guided inference prompt
│   └── skill_redistill.txt     # Lamarckian re-distillation prompt
├── scripts/
│   ├── 2_generate_traces.py    # Step 1: Generate reasoning traces
│   ├── 3_verify_answers.py     # Step 2: Verify trace correctness
│   ├── 4_distill_skills.py     # Step 3: Distill skills from traces
│   ├── 5_retrieve_and_infer.py # Step 4: Skill retrieval + inference
│   ├── filter_and_merge_skills.py  # Self-evaluation & filtering
│   ├── baseline_no_think.py    # No-skill baseline inference
│   └── 7_evolve_skills.py      # (Optional) Lamarckian evolution
├── src/
│   ├── api.py                  # Async LLM client
│   ├── data.py                 # Data I/O, answer matching
│   └── retrieval.py            # SkillBank (BM25 + embedding)
└── outputs/                    # Experiment outputs
```

## Skill Format

Each skill in the skill bank is a JSONL record:

```json
{
  "question_id": "OptMATH_Train_0042",
  "question": "Original training problem text...",
  "answer": "42.0",
  "procedure": "Step 1: Define decision variables...\nStep 2: ...",
  "worked_example": "Let x1, x2 be production quantities...",
  "keywords": "linear programming, production planning, capacity constraints",
  "inject_text": "Step 1: Define decision variables...\n\nWorked Example:\nLet x1, x2..."
}
```

## Benchmarks

| Dataset | Domain | Size |
|---------|--------|------|
| OptMATH_Bench_166 | Mixed OR (scheduling, LP, facility, transport) | 166 |
| MAMO_ComplexLP | Complex Linear Programming | 203 |
| IndustryOR | Industrial OR Problems | 100 |
| NL4OPT | Natural Language to Optimization | 245 |
| OptiBench | Optimization Benchmark | 605 |
| MILP_eval | Mixed-Integer Linear Programming | 360 |
