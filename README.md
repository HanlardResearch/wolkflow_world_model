# 世界模型说明书

本文档对应当前仓库中的世界模型实现，重点解释以下 5 件事：

1. 当前世界模型到底建模什么。
2. 在线推理数据如何变成离线 JSONL。
3. `graph / next_graph / reward / returns / credit_targets` 的最新语义。
4. 训练脚本、模型输入、损失函数如何对齐这些数据。
5. 当前版本已经修掉了什么，还剩哪些已知限制。

本文档基于当前代码版本编写。下面引用的行号均以当前仓库为准，后续代码改动后可能漂移。

---

## 0. 一页总览

当前 `WorkflowWorldModel` 不是纯文本语言模型，而是一个面向多智能体 workflow 的结构化一步世界模型。

它学习的对象可以概括为：

```text
(task, graph_t, state_t, action_t)
  -> encode current observation
  -> predict next latent
  -> predict reward / cost / done / value / uncertainty / valid_action / aux
```

但这里最关键的变化是：

- 在线策略内部仍然保留一套 RL 风格的原始 `trajectory[i]["reward"]`，用于原系统的调度逻辑。
- 真正写进世界模型数据集的 `outcome.reward / returns / credit_targets`，现在优先使用树回传得到的 `world_model_*` 字段，而不是直接使用原始 shaping reward。对应实现见 `puppeteer/inference/policy/workflow_dataset_recorder.py:182`、`193`、`205`、`210`、`377`。

换句话说，当前世界模型训练的主要目标，已经从“直接拟合原始逐步负成本 reward”改成了“拟合基于多分支终局结果回传得到的过程价值和信用分配信号”。

---

## 1. 当前端到端链路

### 1.1 在线推理结束后的入口

任务结束时，`GraphReasoning.finalize()` 不再对每条 path 逐条调用 `finalize_task()`，而是：

1. 对每条 path 计算终局 `transition["reward"]`
2. 先把所有 path 的 `(transition, global_info)` 收集起来
3. 一次性交给 `self.policy.finalize_task_batch(...)`

对应代码：

- `puppeteer/inference/reasoning/reasoning.py:193`
- `puppeteer/inference/reasoning/reasoning.py:255`

### 1.2 Policy 层的新职责

`ContinuousREINFORCE` 当前承担两条链路：

1. 在线调度链
   - `init_forward()` `puppeteer/inference/policy/REINFORCE_continuous.py:1326`
   - `iter_forward()` `puppeteer/inference/policy/REINFORCE_continuous.py:1363`
   - `append_to_trajectory()` `puppeteer/inference/policy/REINFORCE_continuous.py:1400`

2. 离线标签构造链
   - `finalize_task_batch()` `puppeteer/inference/policy/REINFORCE_continuous.py:1530`
   - `_prepare_trajectory_for_finalize()` `puppeteer/inference/policy/REINFORCE_continuous.py:1576`
   - `_build_world_model_tree_targets()` `puppeteer/inference/policy/REINFORCE_continuous.py:1627`
   - `_annotate_trajectory_with_world_model_targets()` `puppeteer/inference/policy/REINFORCE_continuous.py:1701`

### 1.3 Recorder 的当前角色

`WorkflowDatasetRecorder` 的职责仍然是把在线执行过程整理成 step-level JSONL，但它现在会优先消费 `world_model_*` override：

- 决策前状态采集：`capture_decision_state()` `puppeteer/inference/policy/workflow_dataset_recorder.py:75`
- 记录完整 trajectory：`record_completed_trajectory()` `puppeteer/inference/policy/workflow_dataset_recorder.py:91`
- override 读取逻辑：`_step_world_model_target()` `puppeteer/inference/policy/workflow_dataset_recorder.py:377`

### 1.4 训练链

离线训练链保持不变：

- 训练脚本入口：`puppeteer/train_workflow_world_model.py:700`
- `next_batch` 构造：`puppeteer/train_workflow_world_model.py:188`
- Adapter 建 batch：`puppeteer/inference/policy/workflow_world_model.py:982`
- 模型损失：`puppeteer/inference/policy/workflow_world_model.py:780`

---

## 2. 当前数据语义的关键变化

这一节最重要。

### 2.1 `graph` 和 `next_graph`

当前定义已经固定为：

```text
graph_t      = 当前动作执行前(prefix[:t])的局部图
next_graph_t = 当前动作执行后(prefix[:t+1])的局部图
```

对应代码：

- `record_completed_trajectory()` 中构造 `pre_graph` / `post_graph`
  - `puppeteer/inference/policy/workflow_dataset_recorder.py:111-141`
- 图快照函数
  - `puppeteer/inference/policy/workflow_dataset_recorder.py:540`

这意味着：

- `graph` 是 pre-action 图
- `next_graph` 是 post-action 图
- 两者都是 path-local、prefix-local 的动态图
- 不再从共享 `agent_graph` 直接回填，也不再混入别的 path 的边

#### 边的方向

当前边方向是实际执行顺序：

```text
上一步 agent -> 当前 agent
```

对应代码：

- `puppeteer/inference/policy/workflow_dataset_recorder.py:549-558`

#### 一个具体例子

若某条 path 的 agent 序列是：

```text
ReasoningAgent -> ConcluderAgent -> TerminatorAgent
```

则：

- `t=0` 时：
  - `graph.edges = []`
  - `next_graph.edges = []`
- `t=1` 时：
  - `graph.edges = []`
  - `next_graph.edges = [["ReasoningAgent_gpt4o", "ConcluderAgent_gpt4o"]]`
- `t=2` 时：
  - `graph.edges = [["ReasoningAgent_gpt4o", "ConcluderAgent_gpt4o"]]`
  - `next_graph.edges = [["ReasoningAgent_gpt4o", "ConcluderAgent_gpt4o"], ["ConcluderAgent_gpt4o", "TerminatorAgent"]]`

### 2.2 `graph.node_stats`

当前 `node_stats` 已经改成当前 path、当前前缀上的局部统计，不再跨任务累积。

对应代码：

- `puppeteer/inference/policy/workflow_dataset_recorder.py:562-610`

具体含义：

- `usage_count`
  - 该 agent 在当前前缀出现了多少次
- `success_rate`
  - 当前前缀内该 agent 的成功率
- `avg_cost`
  - 当前前缀内该 agent 的平均实际 cost
- `avg_reward`
  - 优先取 `world_model_reward`，否则回退到原始 `reward`
- `avg_credit`
  - 优先取 `world_model_step_credit`，否则回退到 `avg_reward`

因此当前 `graph.node_stats` 的语义已经变成“路径局部图状态”，而不再是“整个采样历史的共享统计”。

### 2.3 `episode_id`

`episode_id` 现在不再包含 `path_id`，而是按任务级别构造：

```text
hash(workpath | task_type | question)
```

对应代码：

- `puppeteer/inference/policy/workflow_dataset_recorder.py:714`

这意味着同一题的 sibling paths 会共享同一个 `episode_id`，便于：

- 训练/验证按题切分
- 避免同题不同 path 分到 train 和 val 两边

`path_id` 仍然保留，用于区分并行分支。

---

## 3. Reward、Return、Credit 的最新定义

### 3.1 终局叶子值 `G(path)`

树回传使用的叶子值来自 `GraphReasoning.finalize()` 里 evaluator 计算出的任务级终局 reward：

- `MMLU-Pro` / `GSM-Hard`
  - 正确 `+1`
  - 错误 `-1`
- `SRDD` / `CW`
  - 使用 evaluator 返回的连续值

对应代码：

- `puppeteer/inference/reasoning/reasoning.py:199-250`

注意：

- 树回传使用的是 `transition["reward"]`
- 不是 `_prepare_trajectory_for_finalize()` 中生成的 shaped terminal reward

也就是说，当前 world-model reward 的叶子值主要反映任务终局质量，而不是原始 RL cost shaping。

### 3.2 原始 trajectory reward 仍然存在

`append_to_trajectory()` 里仍然会先写原始 shaping reward：

- `puppeteer/inference/policy/REINFORCE_continuous.py:1400`

其来源仍是：

```text
logarithmic_cost(step_index) * agent_reward_factor[action]
```

并且在 `_prepare_trajectory_for_finalize()` 中，非终止步 reward 还会继续乘真实 `action.cost / 100000`，终止步还会按旧逻辑生成 shaped terminal reward。

对应代码：

- `puppeteer/inference/policy/REINFORCE_continuous.py:1576`

这套原始 reward 仍服务于原策略轨迹本身，但 recorder 现在优先使用树回传 override，因此世界模型训练通常不会直接拟合这套混合 reward。

### 3.3 树回传的节点值和边值

当前实现使用最简单、最稳的均值回传：

#### 前缀节点值

```text
V(prefix) = 该 prefix 下所有叶子终局 reward 的平均值
```

#### 选中边的 action value

```text
Q(prefix, action) = V(prefix + action)
```

#### 当前 step 的世界模型 reward

```text
outcome.reward = Q(prefix, action)
```

#### 当前 step 的 MC return

```text
returns.mc_return = Q(prefix, action)
```

#### 当前 step 的 step credit

```text
credit_targets.step_credit = Q(prefix, action) - V(prefix)
```

#### 当前 step 的 leave-one-out gap

若该父节点存在兄弟分支：

```text
credit_targets.leave_one_out_gap =
    Q(prefix, action) - mean(V(other_siblings))
```

若没有兄弟分支，则退化为：

```text
credit_targets.leave_one_out_gap = credit_targets.step_credit
```

#### 当前 step 的 `h2_return`

当前实现不是两步 reward 和，而是两步 `step_credit` 折扣和：

```text
h2_t = step_credit_t + gamma * step_credit_{t+1}
```

以上逻辑全部在：

- `puppeteer/inference/policy/REINFORCE_continuous.py:1627-1698`

### 3.4 这些量如何写进 JSONL

`WorkflowDatasetRecorder` 当前会优先读取以下 override：

- `world_model_reward`
- `world_model_mc_return`
- `world_model_h2_return`
- `world_model_step_credit`
- `world_model_leave_one_out_gap`

对应代码：

- `puppeteer/inference/policy/workflow_dataset_recorder.py:182-212`
- synthetic terminator 同样适用：`puppeteer/inference/policy/workflow_dataset_recorder.py:284-314`
- fallback 逻辑：`puppeteer/inference/policy/workflow_dataset_recorder.py:377`

因此当前数据集中：

- `outcome.reward`
  - 主要语义是树回传的 `action_value`
- `returns.mc_return`
  - 当前实现下通常与 `outcome.reward` 同义
- `credit_targets.step_credit`
  - 是相对父节点的增量价值
- `credit_targets.leave_one_out_gap`
  - 是相对兄弟分支基线的优势

### 3.5 额外写入的树回传元数据

当前记录里还会在 `metadata` 下保留调试字段：

- `tree_parent_value`
- `tree_action_value`
- `tree_sibling_baseline`
- `tree_descendant_leaves`
- `tree_leaf_reward`

对应代码：

- `puppeteer/inference/policy/workflow_dataset_recorder.py:222-238`
- synthetic terminator：`puppeteer/inference/policy/workflow_dataset_recorder.py:320-336`

---

## 4. 当前 JSONL 样本的结构语义

### 4.1 顶层结构

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

对应写入函数：

- `puppeteer/inference/policy/workflow_dataset_recorder.py:91`

### 4.2 `state` / `next_state`

`state` 和 `next_state` 都由 `_build_state_snapshot()` 生成：

- `puppeteer/inference/policy/workflow_dataset_recorder.py:384`

其内部使用 `_build_workflow_summary()`：

- `puppeteer/inference/policy/workflow_dataset_recorder.py:422`

当前 `state` 的核心字段包括：

- `workflow_state`
- `executed_steps`
- `recent_answers`
- `reasoning_results`
- `tool_results`
- `all_actions`
- `valid_actions`
- `budget`
- `workflow_valid_actions`
- `path_id`

### 4.3 synthetic terminator

若 trajectory 末尾是 policy 人工补出的 `TerminatorAgent`，且 `include_synthetic_termination=true`，则 recorder 会额外写一个终止样本。

对应代码：

- `puppeteer/inference/policy/workflow_dataset_recorder.py:252-342`

因此单条 path 的 JSONL 条数可能是：

- 实际 workflow 步数
- 或者 实际 workflow 步数 + 1 个 synthetic termination step

---

## 5. Scheduler 采样多样性的最新逻辑

当前采样不再只按“最短正确路径”强行收缩，而是显式增加了早期和不确定阶段的分支保留。

### 5.1 Prompt 层

`_build_messages()` 当前会明确告诉 scheduler：

- 早期阶段优先保留互补 agent
- 证据不全、冲突存在时不要过早塌缩到单路径
- 多 agent 时避免返回同质备份

对应代码：

- `puppeteer/inference/policy/REINFORCE_continuous.py:645`

### 5.2 keep-k 逻辑

当前的 keep-k 决策分成三层：

1. 基础置信度裁剪
   - `_determine_scheduler_keep_k()`
   - `puppeteer/inference/policy/REINFORCE_continuous.py:1133`
2. 最小多样性分支数
   - `_minimum_scheduler_keep_k()`
   - `puppeteer/inference/policy/REINFORCE_continuous.py:1162`
3. 互补分支选择
   - `_select_diverse_scheduler_decisions()`
   - `puppeteer/inference/policy/REINFORCE_continuous.py:1230`

`_minimum_scheduler_keep_k()` 会在以下情况下抬高最小分支数：

- 当前处于很早的 step
- reasoning / external evidence / conclusion 尚不完整
- workflow summary 显示冲突

`_select_diverse_scheduler_decisions()` 会额外偏好：

- 不同 `best_for_stage`
- 是否需要外部工具的差异
- 不同 `output_type`

同时对非必要的 `TerminatorAgent` 做惩罚，避免过早终止。

### 5.3 当前默认配置

配置见 `puppeteer/config/policy.json`：

- `top1_activation_threshold = 0.72`
- `top1_margin_threshold = 0.28`
- `top2_cumulative_threshold = 0.88`
- `top3_cumulative_threshold = 0.97`
- `diversity_enabled = true`
- `diversity_min_keep_k = 2`
- `conflict_min_keep_k = 3`
- `diversity_early_step = 2`

如果你觉得路径仍然过于单一，优先调这里。

---

## 6. 世界模型训练如何读取这些字段

### 6.1 训练脚本

训练脚本入口：

- `puppeteer/train_workflow_world_model.py:700`

关键步骤：

- 参数解析：`puppeteer/train_workflow_world_model.py:26`
- 数据文件发现：`puppeteer/train_workflow_world_model.py:107`
- 按 episode 切 train/val：`puppeteer/train_workflow_world_model.py:164`
- `next_batch` 构造：`puppeteer/train_workflow_world_model.py:188`
- 训练一个 epoch：`puppeteer/train_workflow_world_model.py:613`
- 验证一个 epoch：`puppeteer/train_workflow_world_model.py:646`

### 6.2 `WorkflowStateAdapter` 的 target 映射

Adapter 入口：

- `puppeteer/inference/policy/workflow_world_model.py:848`
- `build_batch()`：`puppeteer/inference/policy/workflow_world_model.py:982`

当前 target 映射在：

- `puppeteer/inference/policy/workflow_world_model.py:1151-1160`

具体为：

- `reward <- outcome.reward`
- `cost <- normalize(outcome.cost_delta)`
- `done <- outcome.done`
- `value <- returns.mc_return`
- `uncertainty <- next_state_targets.conflict_score`
- `counterfactual_credit <- credit_targets.leave_one_out_gap`

因此：

- `reward` 头现在主要在学树回传的 `action_value`
- `value` 头当前实现下通常也在学同一量纲的值
- `counterfactual` 头在学“相对兄弟基线的优势”

### 6.3 模型结构

关键入口：

- `WorkflowWorldModelConfig` `puppeteer/inference/policy/workflow_world_model.py:224`
- `WorkflowWorldModelTargets` `puppeteer/inference/policy/workflow_world_model.py:257`
- `WorkflowWorldModelBatch` `puppeteer/inference/policy/workflow_world_model.py:284`
- `SequenceEncoder` `puppeteer/inference/policy/workflow_world_model.py:385`
- `SetEncoder` `puppeteer/inference/policy/workflow_world_model.py:421`
- `GraphEncoder` `puppeteer/inference/policy/workflow_world_model.py:436`
- `HFTextEncoder` `puppeteer/inference/policy/workflow_world_model.py:465`
- `WorkflowWorldModel` `puppeteer/inference/policy/workflow_world_model.py:520`
- `encode_observation()` `puppeteer/inference/policy/workflow_world_model.py:654`
- `imagine_rollout()` `puppeteer/inference/policy/workflow_world_model.py:738`
- `q_value()` `puppeteer/inference/policy/workflow_world_model.py:776`
- `compute_losses()` `puppeteer/inference/policy/workflow_world_model.py:780`

### 6.4 损失函数

当前损失仍是多任务组合，关键行号：

- `reward` loss `puppeteer/inference/policy/workflow_world_model.py:810`
- `cost` loss `puppeteer/inference/policy/workflow_world_model.py:813`
- `done` loss `puppeteer/inference/policy/workflow_world_model.py:816`
- `value` loss `puppeteer/inference/policy/workflow_world_model.py:819`
- `uncertainty` loss `puppeteer/inference/policy/workflow_world_model.py:822`
- `valid` loss `puppeteer/inference/policy/workflow_world_model.py:826`
- `aux` loss `puppeteer/inference/policy/workflow_world_model.py:834`
- `counterfactual` loss `puppeteer/inference/policy/workflow_world_model.py:843`

`q_value()` 仍定义为：

```text
reward - cost + gamma * value
```

对应：

- `puppeteer/inference/policy/workflow_world_model.py:776`

---

## 7. 当前版本已经修掉的旧问题

以下问题已经不是当前版本的真实状态：

1. `graph.edges` 混入别的 path
   - 已修复
   - 当前为 path-local prefix graph
2. `edges` 方向与执行顺序相反
   - 已修复
   - 当前是 `previous_agent -> current_agent`
3. `next_graph` 基本等于 `graph`
   - 已修复
   - 当前为 pre-action / post-action 两个不同前缀图
4. `episode_id` 以 `path_id` 为主
   - 已修复
   - 当前 sibling paths 共享同一 episode
5. `leave_one_out_gap == mc_return`
   - 已修复
   - 当前已变成相对兄弟基线的优势
6. `node_stats` 跨任务累计
   - 已修复
   - 当前只依赖当前 path 的当前前缀

---

## 8. 当前仍然存在的限制

这些问题仍然存在，文档必须明确：

### 8.1 `valid_action_mask` 仍然基本是常量

`next_state_targets.valid_action_mask` 当前仍直接写成：

```text
list(self.agent_role_list)
```

对应代码：

- `puppeteer/inference/policy/workflow_dataset_recorder.py:537`

这意味着 valid 头仍然偏弱，`valid_f1` 等指标要谨慎解释。

### 8.2 `uncertainty` 与 `conflict_score` 仍然同源

Adapter 当前仍然写：

- `uncertainty <- next_state_targets.conflict_score`

对应代码：

- `puppeteer/inference/policy/workflow_world_model.py:1155`

因此 `uncertainty` 头和 `aux_conflict_score` 头仍然语义重叠。

### 8.3 单路径任务的树信用信号会退化

如果某个任务最终只保留了一条 path，则：

- 没有兄弟分支
- `leave_one_out_gap` 会退化为 `step_credit`

这不是 bug，但代表 counterfactual 监督在单路径样本上信息量较小。

### 8.4 `done` 仍可能学到 Terminator shortcut

`action_features` 里仍显式包含：

- `1.0 if "Terminator" in action_name else 0.0`

因此 `done` 头可能仍会利用 Terminator 这一捷径。

---

## 9. 建议阅读顺序

如果你要继续改这一套系统，建议按下面顺序读代码：

1. `puppeteer/inference/reasoning/reasoning.py:193`
2. `puppeteer/inference/policy/REINFORCE_continuous.py:1530`
3. `puppeteer/inference/policy/workflow_dataset_recorder.py:91`
4. `puppeteer/inference/policy/workflow_world_model.py:982`
5. `puppeteer/inference/policy/workflow_world_model.py:780`
6. `puppeteer/train_workflow_world_model.py:700`

如果你只想排查数据标签问题，优先看：

1. `puppeteer/inference/policy/REINFORCE_continuous.py:1627`
2. `puppeteer/inference/policy/workflow_dataset_recorder.py:182`
3. `puppeteer/inference/policy/workflow_world_model.py:1151`

---

## 10. 最后一句结论

当前版本的世界模型数据链路，已经从“直接复用原始逐步负 reward”升级为：

```text
多路径终局结果
  -> 共享前缀树回传
  -> action value / parent value / sibling baseline
  -> step-level JSONL
  -> world model targets
```

这使得当前数据集更适合训练：

- 过程价值预测
- 边际信用分配
- 多路径规划与 rerank

但它仍然不是一个彻底完成版系统。下一步最值得继续改的，仍然是：

1. 真实的 `valid_action_mask`
2. 更独立的 uncertainty 标注
3. 更强的 counterfactual / rollout 监督
