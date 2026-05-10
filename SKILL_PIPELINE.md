# Code Generation Skill Pipeline — Operator Guide

> 本文档供 AI 助手（Claude/Ducc）读取并执行 code generation skill 自进化 pipeline。
> 当用户说"帮我跑 skill 生成流程"或"重新生成 skill"时，按此文档执行。

---

## Pipeline 概述

Skill 自进化 pipeline 从训练数据出发，生成可复用的 solving procedure + code template，用于提升模型的 OR 问题求解能力。

```
训练数据 → [Step 1] Think-mode 推理 → [Step 2] 验证答案 → [Step 3] 蒸馏 Skill
→ [Step 4] LOO 评估 → [Step 5] 过滤合并 → [Step 6] 测试集验证
```

每个 skill 包含：`solving_procedure`（求解步骤）、`worked_example`（示例计算）、`code_template`（Python 代码模板）、`retrieval_keywords`（检索关键词）。

---

## 前置条件

### 1. vLLM 服务

需要 1-3 个 vLLM 服务（端口 8001/8002/8003），用于并行加速。

```bash
# 启动示例（Qwen3-14B, TP=2）
CUDA_VISIBLE_DEVICES=0,1 python3 -m vllm.entrypoints.openai.api_server \
  --model model/Qwen3-14B --port 8001 \
  --tensor-parallel-size 2 --max-model-len 32768 \
  --trust-remote-code --enable-reasoning --reasoning-parser deepseek_r1 \
  --gpu-memory-utilization 0.9 &

# 验证
curl -s http://127.0.0.1:8001/health
```

`--enable-reasoning --reasoning-parser deepseek_r1` 是 Step 1 Think-mode 必需的。

### 2. 训练数据

JSONL 格式，每行字段：
```json
{"question_id": "xxx", "benchmark": "xxx", "question": "问题文本", "answer": "数值答案"}
```

**现有训练数据位置**：

| 数据集 | 路径 | 条数 |
|--------|------|------|
| OptiBench | `data/benchmarks/augmented_train/OptiBench_aug_train.jsonl` | 1360 |
| IndustryOR | `data/benchmarks/augmented_train/IndustryOR_aug_train.jsonl` | 1051 |
| EasyLP | `data/benchmarks/augmented_train/EasyLP_aug_train.jsonl` | 1200 |
| NL4OPT | `data/benchmarks/augmented_train/NL4OPT_aug_train.jsonl` | 931 |
| NLP4LP | `data/benchmarks/augmented_train/NLP4LP_aug_train.jsonl` | 1237 |
| OptMATH | `data/benchmarks/augmented_train/OptMATH_aug_train.jsonl` | 778 |
| ComplexLP | `data/benchmarks/augmented_train/ComplexLP_aug_train.jsonl` | 544 |

旧版训练数据在 `data/train_data/` 目录。

### 3. 测试数据

| Key | 测试文件 | 条数 |
|-----|---------|------|
| optibench | `data/benchmarks/OptiBench.jsonl` | 605 |
| industryor | `data/benchmarks/IndustryOR_fixedV2.jsonl` | 100 |
| easylp | `data/benchmarks/MAMO_EasyLP_fixed.jsonl` | 642 |
| nl4opt | `data/benchmarks/NL4OPT_Test.jsonl` | 245 |
| nlp4lp | `data/benchmarks/NLP4LP_full.jsonl` | 322 |
| optmath | `data/benchmarks/OptMATH_Bench_166.jsonl` | 166 |
| complexlp | `data/benchmarks/MAMO_ComplexLP_fixed.jsonl` | 203 |

### 4. Config 文件

每个 vLLM 端口一个 config，包含三个 section：

```yaml
model: "model/Qwen3-14B"
api_base_url: "http://127.0.0.1:800X/v1/chat/completions"
api_key: "dummy"

trace_generation:          # Step 1 用
  temperature: 0.7
  max_tokens: 16384
  max_concurrent: 16
  max_retries: 3
  timeout: 600

skill_distillation:        # Step 3 用
  temperature: 0.3
  max_tokens: 4096
  max_concurrent: 16
  max_retries: 3
  timeout: 300

inference:                 # Step 4/6 用
  temperature: 0.0
  max_tokens: 16384
  max_concurrent: 16
  max_retries: 3
  timeout: 600
  top_k: 2
  no_think: true
  retrieval:
    method: structural_hybrid
    embed_model: BAAI/bge-small-en-v1.5
    rrf_k: 60
    min_score: 0
    top1_ratio: 999
    confidence_threshold: 0
    ratio_cutoff: 0
```

**现有 config**：`configs/aug_pipeline_port{1,2,3}.yaml`（分别对应端口 8001/8002/8003）

---

## 一键运行

最简单的方式是用统一脚本跑全部 7 个 benchmark：

```bash
# 确保 3 个 vLLM 服务都在运行
nohup bash scripts/run_aug_pipeline_14b.sh > outputs/aug_pipeline_14b/main.log 2>&1 &
```

该脚本按端口分组并行：
- Port 1: OptiBench + IndustryOR
- Port 2: EasyLP + NL4OPT
- Port 3: NLP4LP + OptMATH + ComplexLP

输出目录：`outputs/aug_pipeline_14b/<benchmark>/`

---

## 分步运行

如果需要单独运行某个步骤或某个 benchmark，以下是每一步的命令。

### Step 1: 生成 Think-mode 推理链

```bash
python3 scripts/2_generate_traces.py \
  <TRAIN_FILE> \
  prompts/source_cot.txt \
  <OUT_DIR>/traces.jsonl \
  --config <CONFIG>
```

- 用 Think mode（`enable_thinking=True`）生成详细推理链
- 提示词 `prompts/source_cot.txt`：要求纯分析推理，不写代码
- **耗时最长**：每题 ~45-90s，16 并发，1000 题约 1-3 小时
- 输出字段：`question_id, question, answer, model_think, model_response, reasoning_trace`

### Step 2: 验证答案正确性

```bash
python3 scripts/3_verify_answers.py \
  <OUT_DIR>/traces.jsonl \
  <OUT_DIR>/verified_traces.jsonl
```

- 从推理链中提取答案（文本提取 + 代码执行 4 种策略）
- 与 gold answer 对比（1% 相对容差）
- 仅保留答案正确的 traces
- 不需要 LLM 调用，纯本地计算，秒级完成
- 典型通过率：40-70%

### Step 3: 蒸馏 Skill

```bash
python3 scripts/4_distill_skills.py \
  <OUT_DIR>/verified_traces.jsonl \
  prompts/skill_distill.txt \
  <OUT_DIR>/skills_raw.jsonl \
  --config <CONFIG>
```

- 用 LLM 从 verified traces 中提取结构化 skill
- 提示词 `prompts/skill_distill.txt`：指定提取 4 个 XML 标签
- 提取标签：
  - `<solving_procedure>`：通用求解步骤
  - `<worked_example>`：压缩版示例计算
  - `<code_template>`：Python 代码模板（max 30 行），根据问题类型选择 solver：
    - PuLP：标准 LP/MIP
    - pyscipopt：复杂 MIP（callbacks, indicator constraints）
    - itertools + MTZ：TSP / 路径规划
    - networkx：网络流、最短路径
    - scipy.optimize：非线性/二次优化
    - for-loop pattern：多期规划、库存
  - `<retrieval_keywords>`：10-20 个 BM25 检索关键词
- 构建 `inject_text = procedure + worked_example + code_template`
- Config section：`skill_distillation`（temp=0.3, 4K tokens）

### Step 4a: LOO（Leave-One-Out）评估

```bash
python3 scripts/5_retrieve_and_infer.py \
  <TRAIN_FILE> \
  <OUT_DIR>/skills_raw.jsonl \
  prompts/skill_infer.txt \
  <OUT_DIR>/predictions_train_loo.jsonl \
  --config <CONFIG> --exclude-self
```

- 对每个训练题，用 BM25 检索 skill（排除自身 `--exclude-self`）
- 注入 skill 后用 No-Think 模式推理
- 提示词 `prompts/skill_infer.txt`：带 `{SOLVING_HINTS}` 和 `{PROBLEM}` 占位符
- 检索方式：structural_hybrid（BM25 + embedding 融合）
- 输出包含：`predicted_answer, is_correct, retrieved_question_ids, retrieved_scores`

### Step 4b: Baseline 推理

```bash
python3 scripts/baseline_no_think.py \
  <TRAIN_FILE> \
  <OUT_DIR>/predictions_train_baseline.jsonl \
  --config <CONFIG>
```

- 无 skill 注入的 No-Think 基线推理
- 用于 Step 5 中计算 skill 的 help/hurt 效果

### Step 5: 过滤与合并 Skill

```bash
python3 scripts/filter_and_merge_skills.py \
  <OUT_DIR>/skills_raw.jsonl \
  <OUT_DIR>/predictions_train_loo.jsonl \
  <OUT_DIR>/predictions_train_baseline.jsonl \
  <OUT_DIR>/skills_filtered.jsonl \
  --report <OUT_DIR>/filter_report.json
```

- 计算每个 skill 的指标：
  - `retrieval_count`：被检索次数
  - `help_count`：baseline 错 → skill 对（帮助次数）
  - `hurt_count`：baseline 对 → skill 错（伤害次数）
  - `net_score = net_help_rate - hurt_rate`
- **过滤规则**：
  - 删除 `net_score < 0`（有害 skill）
  - 删除 `net_help_rate == 0 && retrieval_count >= 5`（高频无用 skill）
- **合并规则**：
  - 用 `BAAI/bge-small-en-v1.5` embedding 计算相似度
  - 按问题类型分组（TSP, VRP, NetworkFlow, MultiPeriod, Nonlinear, Scheduling, FacilityLocation, Knapsack, StandardLP）
  - 同类型内 cosine > 阈值（0.85-0.92）的 skill 合并为一个 cluster
  - 保留 `net_score` 最高的代表
- `filter_report.json` 包含过滤统计

### Step 6a: 测试集 Skill 推理

```bash
python3 scripts/5_retrieve_and_infer.py \
  <TEST_FILE> \
  <OUT_DIR>/skills_filtered.jsonl \
  prompts/skill_infer.txt \
  <OUT_DIR>/predictions_test_skill.jsonl \
  --config <CONFIG>
```

### Step 6b: 测试集 Baseline 推理

```bash
python3 scripts/baseline_no_think.py \
  <TEST_FILE> \
  <OUT_DIR>/predictions_test_baseline.jsonl \
  --config <CONFIG>
```

---

## 监控进度

```bash
# 检查文件生成进度
for bench in optibench industryor easylp nl4opt nlp4lp optmath complexlp; do
  dir="outputs/aug_pipeline_14b/$bench"
  echo -n "$bench: "
  for f in traces.jsonl verified_traces.jsonl skills_raw.jsonl \
           predictions_train_loo.jsonl predictions_train_baseline.jsonl \
           skills_filtered.jsonl predictions_test_skill.jsonl predictions_test_baseline.jsonl; do
    [ -f "$dir/$f" ] && echo -n " $f($(wc -l < "$dir/$f"))"
  done
  echo ""
done

# 检查进程
ps aux | grep -E "2_generate_traces|3_verify|4_distill|5_retrieve|baseline_no_think|filter_and_merge" | grep -v grep

# 查看日志
tail -20 outputs/aug_pipeline_14b/port{1,2,3}.log
```

**时间估算**（Qwen3-14B, 16 并发）：
- Step 1（trace gen）: ~1-5 小时/批（瓶颈步骤）
- Step 2（verify）: ~1 分钟
- Step 3（distill）: ~30-60 分钟/批
- Step 4a（LOO）: ~30-60 分钟/批
- Step 4b（baseline）: ~30-60 分钟/批
- Step 5（filter）: ~2-5 分钟
- Step 6a/6b（test）: ~10-30 分钟

---

## 查看结果

```python
python3 << 'PYEOF'
import json, sys, os
sys.path.insert(0, 'src')
from data import answers_match

base = "outputs/aug_pipeline_14b"
benchmarks = [
    ("OptiBench",  "optibench"),
    ("IndustryOR", "industryor"),
    ("EasyLP",     "easylp"),
    ("NL4OPT",    "nl4opt"),
    ("NLP4LP",    "nlp4lp"),
    ("OptMATH",   "optmath"),
    ("ComplexLP",  "complexlp"),
]

header = f"{'Benchmark':<12} | {'Skill':>10} | {'Baseline':>10} | {'Delta':>7} | {'Skills#':>7}"
print(header)
print("-" * len(header))

for name, d in benchmarks:
    sp = f"{base}/{d}/predictions_test_skill.jsonl"
    bp = f"{base}/{d}/predictions_test_baseline.jsonl"
    if not os.path.exists(sp) or not os.path.exists(bp):
        print(f"{name:<12} |        N/A |        N/A |     N/A |     N/A")
        continue
    def acc(p):
        rows = [json.loads(l) for l in open(p)]
        c = sum(1 for r in rows if answers_match(r.get('predicted_answer',''), str(r.get('answer',''))))
        return c, len(rows)
    sc, st = acc(sp)
    bc, bt = acc(bp)
    sa, ba = sc/st*100, bc/bt*100
    sf = sum(1 for _ in open(f"{base}/{d}/skills_filtered.jsonl")) if os.path.exists(f"{base}/{d}/skills_filtered.jsonl") else 0
    print(f"{name:<12} | {sa:8.1f}%  | {ba:8.1f}%  | {sa-ba:+5.1f}%  | {sf:>7}")
PYEOF
```

---

## 自定义运行

### 更换模型

1. 启动新模型的 vLLM 服务
2. 创建新 config（修改 `model` 和 `api_base_url`）
3. 创建新输出目录运行 pipeline

### 更换训练数据

1. 准备 JSONL 格式数据（必须有 `question_id, question, answer` 字段）
2. 修改 pipeline 脚本中的 `TRAIN_FILE` 路径
3. 使用新输出目录（避免覆盖旧结果）

### 更换蒸馏提示词

修改 `prompts/skill_distill.txt`。当前版本已包含 solver-specific 的 `<code_template>` 指导。Pipeline 脚本会自动使用最新版本。

### 只重跑某一步

Pipeline 脚本有幂等检查（文件存在则跳过）。要重跑某步：
```bash
# 例如：重跑 Step 3（skill 蒸馏）
rm outputs/aug_pipeline_14b/optibench/skills_raw.jsonl
# 然后重新运行 pipeline 脚本，会从 Step 3 开始
```

注意：删除某步输出后，后续步骤也需要删除才能重跑。

---

## 核心文件索引

| 文件 | 用途 |
|------|------|
| `scripts/2_generate_traces.py` | Step 1: Think-mode 推理链生成 |
| `scripts/3_verify_answers.py` | Step 2: 答案验证（本地） |
| `scripts/4_distill_skills.py` | Step 3: Skill 蒸馏（提取 4 个 XML 标签） |
| `scripts/5_retrieve_and_infer.py` | Step 4a/6a: BM25 检索 + Skill 推理 |
| `scripts/baseline_no_think.py` | Step 4b/6b: 无 Skill 基线推理 |
| `scripts/filter_and_merge_skills.py` | Step 5: 过滤有害 Skill + 合并相似 Skill |
| `scripts/run_aug_pipeline_14b.sh` | 一键运行全部 7 benchmark |
| `prompts/source_cot.txt` | Step 1 提示词（纯分析，不写代码） |
| `prompts/skill_distill.txt` | Step 3 提示词（提取 procedure + code_template） |
| `prompts/skill_infer.txt` | Step 4/6 提示词（注入 skill 后推理） |
| `configs/aug_pipeline_port{1,2,3}.yaml` | 3 端口配置文件 |
| `src/data.py` | 数据工具（render_prompt, extract_xml_content, answers_match） |
| `src/retrieval.py` | BM25 检索（SkillBank） |
| `src/api.py` | LLM API 调用 |
