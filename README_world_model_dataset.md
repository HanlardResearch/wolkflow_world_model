# Workflow World Model Dataset README

本文档说明当前 `workflow_world_model.jsonl` 的真实语义。重点是最新版本已经改掉了旧数据中的 4 个核心问题：

1. `graph` / `next_graph` 不再是共享图快照，而是当前 path 的前缀局部图。
2. `edges` 方向改成了真实执行顺序。
3. `episode_id` 不再包含 `path_id`，同题 sibling paths 会共享同一 episode。
4. `reward / returns / credit_targets` 默认优先使用树回传得到的 `world_model_*` 字段。

对应代码：

- recorder: `puppeteer/inference/policy/workflow_dataset_recorder.py:91`
- tree backup: `puppeteer/inference/policy/REINFORCE_continuous.py:1627`

---

## 1. 这个数据集是什么

每一行近似对应一个世界模型 transition：

```text
(graph_t, state_t, action_t, graph_{t+1}, state_{t+1}, outcome_t, targets_t)
```

其中：

- `graph_t` / `state_t`
  - 当前动作执行前的观测
- `action_t`
  - 当前实际执行的 agent
- `graph_{t+1}` / `state_{t+1}`
  - 当前动作执行后的真实观测
- `outcome_t`
  - 直接监督头使用的标签
- `targets_t`
  - return、credit、aux 等附加监督

---

## 2. 数据是如何生成的

### 2.1 决策前状态采集

`capture_decision_state(global_info)` 会在 scheduler 决策前记录当前状态：

- `puppeteer/inference/policy/workflow_dataset_recorder.py:75`

这就是严格意义上的 `s_t`。

### 2.2 选中动作后写入 trajectory

`append_to_trajectory()` 会把当前动作和策略上下文写到 trajectory：

- `puppeteer/inference/policy/REINFORCE_continuous.py:1400`

关键字段：

- `action`
- `reward`
- `state_snapshot`
- `candidate_agents`
- `selected_confidence`
- `path_id`

注意这里的 `reward` 仍是原始策略 reward，不是最终 world-model reward。

### 2.3 任务结束后统一树回传

`GraphReasoning.finalize()` 会把所有 path 的终局 transition 收集起来，再调用：

- `self.policy.finalize_task_batch(...)`

对应：

- `puppeteer/inference/reasoning/reasoning.py:193`
- `puppeteer/inference/reasoning/reasoning.py:255`

`finalize_task_batch()` 内部会：

1. 先按旧逻辑补齐 trajectory 末尾和原始 reward
2. 再基于所有 sibling paths 构 prefix tree
3. 用叶子终局 reward 回传出每个 step 的 `world_model_*` 目标

对应：

- `puppeteer/inference/policy/REINFORCE_continuous.py:1530`
- `puppeteer/inference/policy/REINFORCE_continuous.py:1627`
- `puppeteer/inference/policy/REINFORCE_continuous.py:1701`

### 2.4 recorder 落盘

`record_completed_trajectory()` 会优先读取 `world_model_*` 字段并写入 JSONL：

- `puppeteer/inference/policy/workflow_dataset_recorder.py:182-212`
- synthetic terminator 同样支持：`puppeteer/inference/policy/workflow_dataset_recorder.py:284-314`

fallback 逻辑在：

- `puppeteer/inference/policy/workflow_dataset_recorder.py:377`

---

## 3. 文件路径与输出规则

配置入口：

- `puppeteer/config/policy.json`

当前默认配置：

```json
"world_model_dataset": {
  "enabled": true,
  "output_dir": "results/world_model_dataset-llm",
  "use_dataset_subdirs": true,
  "split_by_time": true,
  "time_granularity": "second",
  "filename": "workflow_world_model.jsonl",
  "max_summary_chars": 240,
  "include_synthetic_termination": true
}
```

因此默认输出类似：

```text
results/world_model_dataset-llm/MMLU-Pro/validation/workflow_world_model_YYYYMMDD_HHMMSS.jsonl
```

训练脚本会按文件名前缀查找，因此带时间戳的文件也会被自动发现。对应：

- `puppeteer/train_workflow_world_model.py:26`
- `puppeteer/train_workflow_world_model.py:107`

---

## 4. 顶层结构

单条样本结构如下：

```json
{
  "episode_id": "...",
  "path_id": 0,
  "t": 1,
  "task": {...},
  "graph": {...},
  "state": {...},
  "next_graph": {...},
  "next_state": {...},
  "action": {...},
  "next_state_targets": {...},
  "outcome": {...},
  "returns": {...},
  "credit_targets": {...},
  "metadata": {...}
}
```

---

## 5. 字段语义

### 5.1 `episode_id`

当前定义：

```text
md5(workpath | task_type | question)
```

对应：

- `puppeteer/inference/policy/workflow_dataset_recorder.py:714`

含义：

- 同一题的 sibling paths 共享同一 `episode_id`
- `path_id` 不再参与 hash

### 5.2 `path_id`

当前 path 编号，仍然保留，用于区分并行分支。

### 5.3 `graph`

当前动作执行前的局部图。

定义：

```text
graph = build_graph(prefix[:t])
```

对应：

- `puppeteer/inference/policy/workflow_dataset_recorder.py:111-141`
- `puppeteer/inference/policy/workflow_dataset_recorder.py:540`

当前 `graph` 具有以下性质：

- path-local
- prefix-local
- `edges` 方向为 `previous_agent -> current_agent`
- `node_stats` 只统计当前 path 当前前缀

### 5.4 `next_graph`

当前动作执行后的局部图。

定义：

```text
next_graph = build_graph(prefix[:t+1])
```

这不再是旧版本里“经常等于 `graph`”的共享静态图。

### 5.5 `state`

当前动作执行前的 workflow 状态。

来自：

- `_build_state_snapshot()` `puppeteer/inference/policy/workflow_dataset_recorder.py:384`
- `_build_workflow_summary()` `puppeteer/inference/policy/workflow_dataset_recorder.py:422`

核心字段：

- `workflow_state`
- `executed_steps`
- `recent_answers`
- `reasoning_results`
- `tool_results`
- `budget`
- `workflow_valid_actions`
- `path_id`

### 5.6 `next_state`

当前动作执行后的 workflow 状态。

训练脚本会用它构造 `next_batch`：

- `puppeteer/train_workflow_world_model.py:188`

### 5.7 `action`

当前执行动作，也就是 `a_t`。

字段包括：

- `kind`
- `name`
- `selected_confidence`
- `estimated_cost`
- `candidate_agents`

### 5.8 `next_state_targets`

当前仍是启发式 dense supervision：

- `progress_score`
- `coverage_score`
- `conflict_score`
- `redundancy_score`
- `termination_readiness`
- `valid_action_mask`

对应：

- `puppeteer/inference/policy/workflow_dataset_recorder.py:495`

注意：

- `valid_action_mask` 当前仍直接写成 `list(self.agent_role_list)`，因此 valid 头仍然偏弱。

### 5.9 `outcome`

当前 step 的直接监督标签。

字段：

- `reward`
- `cost_delta`
- `token_delta`
- `done`
- `success`

当前优先级：

```text
outcome.reward =
  world_model_reward if present
  else raw trajectory reward
```

对应：

- `puppeteer/inference/policy/workflow_dataset_recorder.py:182`

在当前主链路中，这个 `reward` 主要表示树回传的 `action_value`。

### 5.10 `returns`

字段：

- `mc_return`
- `h2_return`

当前优先级：

```text
mc_return =
  world_model_mc_return if present
  else discounted raw reward

h2_return =
  world_model_h2_return if present
  else two-step discounted raw reward
```

对应：

- `puppeteer/inference/policy/workflow_dataset_recorder.py:193`
- `puppeteer/inference/policy/workflow_dataset_recorder.py:198`
- fallback 计算：`puppeteer/inference/policy/workflow_dataset_recorder.py:351`、`361`

在当前树回传实现中：

- `mc_return` 通常等于 `action_value`
- `h2_return` 是两步 `step_credit` 的折扣和

### 5.11 `credit_targets`

字段：

- `leave_one_out_gap`
- `step_credit`

当前优先级：

```text
leave_one_out_gap =
  world_model_leave_one_out_gap if present
  else raw_returns[t]

step_credit =
  world_model_step_credit if present
  else raw_reward[t]
```

但在当前主链路下，通常会使用树回传值：

- `step_credit = action_value - parent_value`
- `leave_one_out_gap = action_value - sibling_baseline`

### 5.12 `metadata`

当前除了原有的调试字段，还新增了树回传字段：

- `tree_parent_value`
- `tree_action_value`
- `tree_sibling_baseline`
- `tree_descendant_leaves`
- `tree_leaf_reward`

对应：

- `puppeteer/inference/policy/workflow_dataset_recorder.py:222-238`

这些字段不会直接进入当前模型主输入，但非常适合做离线分析和抽样排查。

---

## 6. 树回传的数学语义

当前实现使用的叶子值来自 evaluator：

```text
G(path) = transition["reward"]
```

节点值：

```text
V(prefix) = mean_{leaf under prefix} G(leaf)
```

选中边值：

```text
Q(prefix, action) = V(prefix + action)
```

写入数据集后的字段对应关系：

- `outcome.reward = Q(prefix, action)`
- `returns.mc_return = Q(prefix, action)`
- `credit_targets.step_credit = Q(prefix, action) - V(prefix)`
- `credit_targets.leave_one_out_gap = Q(prefix, action) - mean(V(other_siblings))`

若没有兄弟分支，则：

- `leave_one_out_gap = step_credit`

对应实现：

- `puppeteer/inference/policy/REINFORCE_continuous.py:1627-1698`

---

## 7. synthetic terminator 的行为

若当前 path 最后一步不是 `TerminatorAgent`，policy 会先补一个 synthetic terminator 到 trajectory：

- `puppeteer/inference/policy/REINFORCE_continuous.py:1602-1625`

随后 recorder 会在 `include_synthetic_termination=true` 时，为它单独写一条终止样本：

- `puppeteer/inference/policy/workflow_dataset_recorder.py:252-342`

因此你在 JSONL 中看到：

- 实际 workflow 只有 2 步
- 但数据里有 3 条记录

这是正常现象。

---

## 8. 训练脚本如何使用这些字段

`WorkflowStateAdapter.build_batch()` 会把样本映射为模型监督目标：

- `reward <- outcome.reward`
- `cost <- normalize(outcome.cost_delta)`
- `done <- outcome.done`
- `value <- returns.mc_return`
- `uncertainty <- next_state_targets.conflict_score`
- `counterfactual_credit <- credit_targets.leave_one_out_gap`

对应：

- `puppeteer/inference/policy/workflow_world_model.py:1151-1160`

这意味着当前模型的几个头主要在学：

- `reward` 头：树回传 action value
- `value` 头：从当前边看出去的节点值
- `counterfactual` 头：相对兄弟基线的优势

---

## 9. 当前版本仍需注意的限制

1. `valid_action_mask` 仍然是常量式标签。
2. `uncertainty` 仍然直接来自 `conflict_score`。
3. 单路径任务上的 `leave_one_out_gap` 会退化成 `step_credit`。
4. recorder 虽然保留了原始 `reward` fallback，但主链路已经不建议再把它当作世界模型主标签解释。

---

## 10. 推荐检查项

抽查数据时，优先检查：

1. `graph` 是否真的是当前动作前的 prefix 图。
2. `next_graph` 是否只比 `graph` 多出当前动作造成的变化。
3. `outcome.reward` 是否已经是树回传的 `action_value`。
4. `credit_targets.step_credit` 是否等于 `action_value - parent_value`。
5. `credit_targets.leave_one_out_gap` 是否正确体现 sibling baseline。
6. 同一题不同 `path_id` 的 `episode_id` 是否相同。

---

## 11. 样例数据结构说明

下面给出一个具体例子，帮助理解当前数据集中 `episode -> paths -> steps` 的关系。

假设当前任务是：

```text
question: "证明 sqrt(2) 是无理数"
episode_id = md5(workpath | task_type | question)
```

则可以把一个 `episode` 理解为“同一道题的一次完整求解任务”，其下可能包含多条 sibling paths：

```text
episode: E_q1
question: "证明 sqrt(2) 是无理数"

├─ path 0
│  path_id = 0
│  ├─ step 0
│  │  state: 还没有执行任何 agent
│  │  action: MathReasonerAgent
│  │  next_state: 生成“反证法”思路
│  │  record: (episode_id=E_q1, path_id=0, t=0)
│  ├─ step 1
│  │  state: 已有反证法思路
│  │  action: ProofWriterAgent
│  │  next_state: 写出证明草稿
│  │  record: (episode_id=E_q1, path_id=0, t=1)
│  └─ step 2
│     state: 已有证明草稿
│     action: TerminatorAgent
│     next_state: done=True
│     record: (episode_id=E_q1, path_id=0, t=2)
│
├─ path 1
│  path_id = 1
│  ├─ step 0
│  │  state: 还没有执行任何 agent
│  │  action: SearchAgent
│  │  next_state: 找到若干参考证明
│  │  record: (episode_id=E_q1, path_id=1, t=0)
│  ├─ step 1
│  │  state: 已有参考证明
│  │  action: MathReasonerAgent
│  │  next_state: 整理出正式论证
│  │  record: (episode_id=E_q1, path_id=1, t=1)
│  ├─ step 2
│  │  state: 已有正式论证
│  │  action: CriticAgent
│  │  next_state: 发现表述不严谨
│  │  record: (episode_id=E_q1, path_id=1, t=2)
│  └─ step 3
│     state: 已修正论证
│     action: TerminatorAgent
│     next_state: done=True
│     record: (episode_id=E_q1, path_id=1, t=3)
│
└─ path 2
   path_id = 2
   ├─ step 0
   │  state: 还没有执行任何 agent
   │  action: MathReasonerAgent
   │  next_state: 生成错误思路
   │  record: (episode_id=E_q1, path_id=2, t=0)
   ├─ step 1
   │  state: 有错误思路
   │  action: CriticAgent
   │  next_state: 指出矛盾，但未修复
   │  record: (episode_id=E_q1, path_id=2, t=1)
   └─ step 2
      state: 仍未完成证明
      action: TerminatorAgent
      next_state: done=True, success=False
      record: (episode_id=E_q1, path_id=2, t=2)
```

对应到当前 JSONL 数据的层级关系，可以这样理解：

- `episode`
  - 表示一道题 / 一个任务实例
  - 同一题下的所有 sibling paths 共享同一个 `episode_id`
- `path`
  - 表示该题下的一条调度探索分支
  - 不同分支用 `path_id` 区分
- `step`
  - 表示该分支上的一次真实动作执行
  - 每个 step 会落成一条 step-level JSONL record

也就是说，数据文件中并不是“一条记录对应一个 episode”，而是：

```text
一个 episode
  -> 包含多条 path
  -> 每条 path 包含多个 step
  -> 每个 step 对应一条 JSONL record
```

如果写成更抽象的形式，就是：

```text
episode
  = {path_0, path_1, path_2, ...}

path_k
  = {step_0, step_1, step_2, ...}

step_t
  = 一个 transition:
    (graph_t, state_t, action_t, next_graph_t, next_state_t, outcome_t, targets_t)
```

因此，当你在构建 world model 数据时：

1. 不要把 `episode` 理解成“单个 step”。
2. 也不要把 `episode` 理解成“单条 path”。
3. 更准确的理解是：`episode` 是“同一道题及其所有并行/分支探索路径”的共同任务单元。


## 附录：Reward/Value 特征审计

本附录基于一个真实样本记录，对当前世界模型的特征流水线进行交叉核对：
`puppeteer/results/world_model_dataset-llm/MMLU-Pro/test/workflow_world_model_20260402_120316.jsonl`。

目标是回答三个问题：

1. 当前哪些字段被用于预测 `reward` 和 `value`？
2. 原始数据集中存在哪些字段，但尚未被使用？
3. 哪些未使用字段值得优先纳入？

### A. 当前已使用的输入特征

在当前 `WorkflowWorldModel` 编码器栈中，以下字段已经被用于预测 `reward` 和 `value`。

| 来源                        | 字段                      | 当前是否使用    | 说明                 |
| ------------------------- | ----------------------- | --------- | ------------------ |
| `task`                    | `type`                  | Yes       | 编码为任务类型/类别。        |
| `task`                    | `Question` / `question` | Yes       | 通过文本统计特征或文本编码器使用。  |
| `state`                   | `workflow_state`        | Yes       | 核心符号状态签名。          |
| `state.executed_steps[*]` | `agent`                 | Yes       | 序列编码器输入。           |
| `state.executed_steps[*]` | `action`                | Yes       | 序列编码器输入。           |
| `state.executed_steps[*]` | `success`               | Yes       | 步级数值特征。            |
| `state.executed_steps[*]` | `tokens`                | Yes       | 步级数值特征。            |
| `state.executed_steps[*]` | `cost`                  | Yes       | 步级数值特征。            |
| `state.executed_steps[*]` | `parameter`             | Partially | 目前仅使用了类似长度这样的浅层特征。 |
| `state.executed_steps[*]` | `answer_summary`        | Partially | 主要使用长度/存在性这类特征。    |
| `state.executed_steps[*]` | `step_data_summary`     | Partially | 主要使用长度/存在性这类特征。    |
| `state`                   | `recent_answers`        | Yes       | 证据/文本集合编码器输入。      |
| `state`                   | `reasoning_results`     | Yes       | 证据/文本集合编码器输入。      |
| `state`                   | `tool_results`          | Yes       | 证据/文本集合编码器输入。      |
| `state.budget`            | `step_index`            | Yes       | 预算特征。              |
| `state.budget`            | `used_tokens`           | Yes       | 预算特征。              |
| `state.budget`            | `used_cost`             | Yes       | 预算特征。              |
| `graph`                   | `nodes`                 | Yes       | 图编码器输入。            |
| `graph`                   | `edges`                 | Yes       | 图编码器输入。            |
| `graph.node_stats[*]`     | `success_rate`          | Yes       | 节点数值特征。            |
| `graph.node_stats[*]`     | `avg_cost`              | Yes       | 节点数值特征。            |
| `graph.node_stats[*]`     | `avg_credit`            | Yes       | 节点数值特征。            |
| `graph.node_stats[*]`     | `usage_count`           | Yes       | 节点数值特征。            |
| `action`                  | `kind`                  | Yes       | 动作编码器输入。           |
| `action`                  | `name`                  | Yes       | 动作编码器输入。           |
| `action`                  | `estimated_cost`        | Yes       | 动作数值特征。            |
| 顶层回退字段                    | `total_tokens`          | Yes       | 在 budget 字段缺失时使用。  |
| 顶层回退字段                    | `total_cost`            | Yes       | 在 budget 字段缺失时使用。  |

### B. 作为监督目标使用，而非作为输入

以下字段被有意设计为监督目标，而不是模型输入。

| 字段                                         | 当前角色                |
| ------------------------------------------ | ------------------- |
| `outcome.reward`                           | `reward` 目标         |
| `outcome.cost_delta`                       | `cost` 目标           |
| `outcome.done`                             | `done` 目标           |
| `returns.mc_return`                        | `value` 目标          |
| `next_state_targets.conflict_score`        | `uncertainty` 目标    |
| `next_state_targets.progress_score`        | 辅助目标                |
| `next_state_targets.coverage_score`        | 辅助目标                |
| `next_state_targets.redundancy_score`      | 辅助目标                |
| `next_state_targets.termination_readiness` | 辅助目标                |
| `next_state_targets.valid_action_mask`     | `valid` 目标          |
| `credit_targets.leave_one_out_gap`         | `counterfactual` 目标 |

### C. 在原始 JSONL 中存在但尚未使用的字段

已在真实样本记录中确认以下字段存在，但当前尚未作为模型输入使用。

| 来源                    | 字段                       | 是否应直接使用？                 | 重要性说明                                  |
| --------------------- | ------------------------ | ------------------------ | -------------------------------------- |
| 顶层                    | `path_id`                | Possibly                 | 有助于区分同一 episode 中的兄弟搜索分支。              |
| 顶层                    | `t`                      | Possibly                 | 显式步索引，不依赖重构历史。                         |
| `task`                | `id`                     | Yes                      | 可作为稳定的样本/题目标识符，用于诊断或数据划分控制。            |
| `task`                | `Answer`                 | No                       | 这会在训练输入中造成标签泄漏。                        |
| `state`               | `all_actions`            | Possibly                 | 可能有助于反映已探索的动作历史。                       |
| `state`               | `valid_actions`          | Not needed now           | 与当前 valid-mask 监督高度重叠，而且通常接近常量。        |
| `state`               | `workflow_valid_actions` | Yes                      | 比通用 valid list 更具体地反映工作流层面的可行动作可用性。    |
| `state`               | `path_id`                | Possibly                 | 可用于识别分支局部上下文。                          |
| `action`              | `selected_confidence`    | Yes                      | 是预测所选动作是否可信的强候选特征。                     |
| `action`              | `candidate_agents`       | Yes                      | 编码的是局部决策前沿，而不仅仅是已选动作。                  |
| `graph.node_stats[*]` | `avg_reward`             | Yes                      | 直接反映每个节点历史动作质量。                        |
| `outcome`             | `token_delta`            | No for online prediction | 属于未来信息；仅适合分析，不应作为当前步输入。                |
| `outcome`             | `success`                | No for online prediction | 属于未来信息；会泄漏标签/结果。                       |
| `returns`             | `h2_return`              | Yes                      | 较短视野的 return 目标可能比完整 `mc_return` 更易学习。 |
| `credit_targets`      | `step_credit`            | Yes                      | 可能比稀疏的 `leave_one_out_gap` 更有信息量。      |
| `metadata`            | `action_parameter`       | Yes                      | 比当前仅用长度处理 step parameters 丰富得多。        |
| `metadata`            | `result_summary`         | Yes                      | 很可能包含区分许多冲突样本的语义信号。                    |
| `metadata`            | `answer_summary`         | Yes                      | 比仅使用浅层长度特征更能直接反映 answer-state 信号。      |
| `metadata`            | `tree_parent_value`      | Possibly                 | 可作为 planner/tree-search 的先验特征。         |
| `metadata`            | `tree_action_value`      | Possibly                 | 可作为 planner/tree-search 的先验特征。         |
| `metadata`            | `tree_sibling_baseline`  | Possibly                 | 有助于将所选分支与其他候选分支进行比较。                   |
| `metadata`            | `tree_descendant_leaves` | Possibly                 | 编码局部搜索深度/分支支持信息。                       |
| `metadata`            | `tree_leaf_reward`       | No for online prediction | 这本质上属于未来结果信息。                          |
| `metadata`            | `metrics`                | Case by case             | 需要检查 schema；其中可能包含有用的结构化诊断信息。          |

### D. 最高优先级的缺失特征

基于冲突分析和真实 JSONL 样本，最有潜力补充进来的输入特征包括：

1. `action.selected_confidence`
2. `action.candidate_agents`
3. `metadata.action_parameter`
4. `metadata.result_summary`
5. `metadata.answer_summary`
6. `graph.node_stats[*].avg_reward`
7. 将 `returns.h2_return` 作为额外的、更易学习的目标
8. 将 `credit_targets.step_credit` 作为比稀疏反事实标签更好的替代项

### E. 核心结论

当前模型并不是缺失了所有特征。它已经使用了任务、状态、历史、证据、图结构、预算和动作信息。

当前的主要缺口其实更集中：一些很可能能够区分冲突性 `reward` / `value` 样本的字段仍然：

* 完全没有被使用，或
* 仅被简化为较弱的浅层特征，例如文本长度 / 存在标记。

这很可能正是 `cost` 和 `uncertainty` 学得较好，而 `reward` 和 `value` 在当前可观测状态定义下仍然存在严重标签冲突的主要原因。


