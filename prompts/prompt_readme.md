# Prompts 文件说明

本文件夹包含 OR-Trace-Skill pipeline 各阶段使用的 prompt 模板。

## 文件列表

### `source_cot.txt`
- **用途**: Step 1 — Think-mode Trace Generation（思维链推理生成）
- **使用脚本**: `scripts/2_generate_traces.py`
- **说明**: 要求模型以数学推理方式逐步解题（不写代码），输出完整的思维链和最终数值答案。用于生成高质量的推理 trace，作为后续 skill 蒸馏的输入。
- **占位符**: `{PROBLEM}` — 待解决的优化问题

### `skill_distill.txt`
- **用途**: Step 3 — Skill Distillation（技能蒸馏）
- **使用脚本**: `scripts/4_distill_skills.py`
- **说明**: 从一个问题+推理过程+正确答案中，提取出可复用的求解程序（solving procedure）、精简的工作示例（worked example）、Python 代码模板（code template）和检索关键词。输出格式为 XML 结构。
- **占位符**: `[INSERT PROBLEM HERE]`, `[INSERT CHAIN OF THOUGHT HERE]`, `[INSERT CORRECT ANSWER HERE]`

### `skill_infer.txt`
- **用途**: Step 4 — LOO Inference（留一法推理评估）
- **使用脚本**: `scripts/5_retrieve_and_infer.py`
- **说明**: 给定检索到的 skill（求解程序和示例），要求模型按照相同步骤解决新问题。用于评估 skill 的有效性（能否帮助模型答对新题）。
- **占位符**: `{SOLVING_HINTS}` — 检索到的 skill 文本, `{PROBLEM}` — 待解决的问题

### `nothink_code.txt`
- **用途**: NoThink+Code 基线评测（无 skill）
- **使用脚本**: `scripts/eval_think_code.py`
- **说明**: 要求模型直接生成可执行的 Python 代码来求解优化问题，不提供任何 skill 提示。用作 baseline 评测和 Step 5 基线推理。
- **占位符**: `{PROBLEM}` — 待解决的问题

### `nothink_skill_code.txt`
- **用途**: NoThink+Skill+Code 评测（带 skill）
- **使用脚本**: `scripts/eval_nothink_skill_code.py`
- **说明**: 在 `nothink_code.txt` 基础上增加了 `{SOLVING_HINTS}` 区域，注入检索到的 skill（建模模式和代码模板）作为参考。用于评估 skill 对代码生成的提升效果。
- **占位符**: `{SOLVING_HINTS}` — 检索到的 skill 文本, `{PROBLEM}` — 待解决的问题

### `synthesize.txt`
- **用途**: 数据合成 — 生成新的训练问题
- **使用脚本**: 数据增强流程
- **说明**: 给定若干示例问题，要求模型生成一个新的、不同的线性规划问题（含正确答案）。用于扩充训练数据集。
- **占位符**: `{EXAMPLES}` — 示例问题列表

## Pipeline 中的使用顺序

```
Step 1: source_cot.txt        → 生成推理 trace
Step 2: (无 prompt，纯验证)    → 验证答案正确性
Step 3: skill_distill.txt     → 从 trace 蒸馏 skill
Step 4: skill_infer.txt       → LOO 评估 skill 有效性
Step 5: nothink_code.txt      → 基线推理（无 skill）
Step 6: (无 prompt，纯过滤)    → 过滤/合并 skills
Step 7: nothink_skill_code.txt → 最终评测（带 skill）
```
