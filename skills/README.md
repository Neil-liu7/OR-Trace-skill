# OR Solver Skills Index

从 Qwen3 think 模式的 thinking trace 中提炼出 5 个核心通用 skill，外加 1 个组合版本。

## 单独使用（可拼接）

| Skill | 文件 | 对应 thinking 中的什么阶段 |
|-------|------|---------------------------|
| **Skill 1: Mathematical Modeling** | `skill_1_mathematical_modeling.md` | 变量定义、目标函数、约束建模 |
| **Skill 2: Systematic Solving** | `skill_2_systematic_solving.md` | 分步代数推导、角点枚举、贪心搜索 |
| **Skill 3: Verification & Correction** | `skill_3_verification_and_correction.md` | 回代验证、自我纠错、答案检查 |
| **Skill 4: Problem Classification** | `skill_4_problem_classification.md` | 问题识别（LP/IP/组合/多期）、策略选择 |
| **Skill 5: Edge Case Analysis** | `skill_5_edge_case_analysis.md` | 边界情况、严格/非严格不等式、单位一致性 |

## 组合使用（推荐）

| 文件 | 说明 |
|------|------|
| **`skill_combined.md`** | 5 个 skill 压缩为一个 5 阶段流程，适合作为 system prompt |

## 实验建议

### 实验 1: Combined skill（推荐先跑）
将 `skill_combined.md` 内容作为 system prompt prefix，在 nothink 模式下测试。

### 实验 2: 消融实验
逐一去除单个 skill，观察哪个 skill 贡献最大：
- No Skill 1 (去掉建模) → 预期影响最大
- No Skill 3 (去掉验证) → 预期中等影响
- No Skill 4 (去掉分类) → 预期对 IndustryOR 影响大

### 实验 3: 最小 skill
只用 Skill 1 (建模)，测试建模本身是否是提升的核心来源。

## 设计理念

从 thinking trace 分析中发现的关键 pattern：

1. **Think 模式的核心行为是"在输出空间中执行求解器"**：模型在 thinking 空间完成 建模→推导→求解→验证 的全流程
2. **NoThink 失败的根因是缺少中间计算空间**：skill prompt 的目标是迫使 nothink 模型在 answer 空间中执行这些步骤
3. **验证/纠错是 think 的独特优势**：think 模式中大量 "Wait, let me re-check..." 式的自我修正，这正是 Skill 3 试图复现的能力
4. **小模型 think trace 更长**：说明 skill 对小模型可能更有帮助（需要更多引导）
