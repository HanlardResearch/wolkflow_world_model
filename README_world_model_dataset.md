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
