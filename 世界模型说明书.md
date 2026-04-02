# 世界模型说明书

本文档面向当前仓库里 `WorkflowWorldModel` 相关代码，目标是帮助你用最短路径系统理解：

1. 世界模型到底建模了什么。
2. 训练数据是怎么从在线执行过程变成离线 JSONL 的。
3. 每个字段、每个模块、每个损失、每个指标在代码里的具体位置和物理含义是什么。
4. 目前这套实现有哪些明显问题、哪些地方最值得优先改。

本文档基于当前仓库版本编写。下面所有“代码位置”均指当前仓库中的相对路径和当前行号，后续代码变动后行号可能漂移。

---

## 0. 一页总览

当前世界模型不是“纯文本语言世界模型”，而是一个**针对多智能体 workflow 执行过程的结构化一步世界模型**。

它学习的对象可以概括为：

```text
(当前任务 + 当前工作流状态 + 历史步骤 + 证据 + 图结构 + 当前动作)
    -> posterior latent
    -> 预测下一步 prior latent
    -> 再预测 reward / cost / done / value / uncertainty / valid_action / aux signals
```

从工程实现看，它由 4 条主链组成：

1. 在线执行链  
   `GraphReasoning.finalize()` 生成任务级 terminal reward。  
   代码入口：`puppeteer/inference/reasoning/reasoning.py:193-269`

2. 轨迹回填链  
   `ContinuousREINFORCE.finalize_task()` 把在线执行的轨迹整理成 step-level reward，并调用 recorder。  
   代码入口：`puppeteer/inference/policy/REINFORCE_continuous.py:1317-1382`

3. 离线数据集构造链  
   `WorkflowDatasetRecorder` 把 `s_t, a_t, s_{t+1}, reward, cost, done, returns, aux` 写成 JSONL。  
   代码入口：`puppeteer/inference/policy/workflow_dataset_recorder.py:35-558`

4. 离线训练链  
   `train_workflow_world_model.py` 读取 JSONL，构造 batch，训练 `WorkflowWorldModel`。  
   代码入口：`puppeteer/train_workflow_world_model.py:26-786`

---

## 1. 建议阅读顺序

如果你的目标是“最快理解全貌”，建议按这个顺序读：

1. 先读本文第 2 节，看端到端数据流。
2. 再读本文第 3 节，看 JSONL 训练样本到底长什么样。
3. 再读本文第 4 节，看 `WorkflowStateAdapter.build_batch()` 如何把 JSONL 变成张量。
4. 再读本文第 5 节，看 `WorkflowWorldModel` 结构。
5. 再读本文第 6 节，看训练目标和损失函数。
6. 最后读本文第 8 节和第 9 节，看指标和当前实现的问题。

如果你的目标是“先定位 bug / 设计缺陷”，建议先看本文第 9 节。

---

## 2. 端到端数据流

### 2.1 从在线推理到离线样本

完整数据流如下：

```text
任务输入 task
  -> GraphReasoning / ContinuousREINFORCE 在线调度 agent
  -> 每次决策前记录 s_t snapshot
  -> 每次选中 agent 后把 action 和基础 reward 写入 trajectory
  -> 任务结束后 evaluator 给 terminal reward
  -> finalize_task() 回填整条 trajectory 的逐步 reward / cost / done
  -> WorkflowDatasetRecorder 写成 step-level JSONL
  -> train_workflow_world_model.py 读取 JSONL
  -> WorkflowStateAdapter.build_batch() 转张量
  -> WorkflowWorldModel 前向 + compute_losses()
```

### 2.2 关键文件地图

| 文件 | 作用 | 关键入口 |
| --- | --- | --- |
| `puppeteer/inference/reasoning/reasoning.py` | 任务结束时聚合答案、打 terminal reward | `GraphReasoning.finalize()` |
| `puppeteer/inference/policy/REINFORCE_continuous.py` | 在线调度与轨迹维护、reward shaping、调用 recorder | `append_to_trajectory()`、`finalize_task()` |
| `puppeteer/inference/policy/workflow_dataset_recorder.py` | 将在线轨迹转换为世界模型训练 JSONL | `capture_decision_state()`、`record_completed_trajectory()` |
| `puppeteer/inference/policy/workflow_world_model.py` | 模型结构、batch adapter、损失函数 | `WorkflowWorldModel`、`WorkflowStateAdapter` |
| `puppeteer/train_workflow_world_model.py` | 训练脚本、指标统计、checkpoint | `main()` |
| `puppeteer/config/policy.json` | 在线策略 reward shaping、world-model dataset 输出配置 | `agent.reward_factors`、`world_model_dataset` |
| `puppeteer/config/global.yaml` | 在线图推理最大步数 | `graph.max_step_num` |

### 2.3 当前世界模型到底预测什么

训练脚本和模型代码显示，这个世界模型主要学 3 类东西：

1. 状态转移  
   用当前观测的 posterior latent 去预测下一时刻的 prior latent。  
   代码：`puppeteer/inference/policy/workflow_world_model.py:629-669`、`748-769`

2. 一步监督头  
   预测 `reward`、`cost`、`done`、`value`、`uncertainty`、`valid_action_logits` 和 `aux`。  
   代码：`puppeteer/inference/policy/workflow_world_model.py:677-706`

3. 用于 planning / rerank 的 rollout / Q 近似  
   `imagine_rollout()` 和 `q_value()`。  
   代码：`puppeteer/inference/policy/workflow_world_model.py:708-746`

---

## 3. 训练数据集是如何构造的

### 3.1 记录器的核心职责

`WorkflowDatasetRecorder` 的职责非常明确：把在线 scheduler / workflow 轨迹变成可训练的一步转移样本。

核心注释和逻辑在：

- `puppeteer/inference/policy/workflow_dataset_recorder.py:11-33`
- `puppeteer/inference/policy/workflow_dataset_recorder.py:85-244`

可以把 recorder 输出理解为：

```text
(s_t, a_t, s_{t+1}, r_t, info_t)
```

这里 `info_t` 里额外放了：

- `cost_delta`
- `done`
- `returns`
- `credit_targets`
- `next_state_targets`
- `metadata`

### 3.2 s_t 是何时记录的

决策前状态 `s_t` 由 `capture_decision_state(global_info)` 记录：

- 入口：`puppeteer/inference/policy/workflow_dataset_recorder.py:71-83`
- 被调用的位置：
  - `puppeteer/inference/policy/REINFORCE_continuous.py:1200-1225`
  - `puppeteer/inference/policy/REINFORCE_continuous.py:1233-1261`

其物理含义是：

- 这是 scheduler 在做动作选择前真正看到的状态。
- 不是事后恢复的，不是未来状态。
- 因此它可以作为世界模型里真正的 `s_t`。

### 3.3 选中动作后，trajectory 里先写了什么

每次选中 agent 之后，`append_to_trajectory()` 会向当前 path 追加一条轨迹项：

代码：`puppeteer/inference/policy/REINFORCE_continuous.py:1265-1297`

写入的关键字段有：

- `prob`
- `log_prob`
- `state_identifier`
- `action`
- `reward`
- `reward_model`
- `state_snapshot`
- `candidate_agents`
- `selected_confidence`
- `path_id`

其中最重要的是：

- `reward`：此时先写入一个**基础 shaping reward**
- `state_snapshot`：动作前状态
- `candidate_agents`：scheduler 的候选 agent 集合
- `selected_confidence`：当前动作被选中时的归一化概率

### 3.4 非终止步 reward 是如何计算的

#### 第一步：先写一个按步数和 agent 类型调制的基础 reward

代码：

- `logarithmic_cost()`：`puppeteer/inference/policy/REINFORCE_continuous.py:977-994`
- `append_to_trajectory()`：`puppeteer/inference/policy/REINFORCE_continuous.py:1277-1289`
- reward factor 配置：`puppeteer/config/policy.json:18-27`

基础 reward 定义为：

```text
r_base(t, a) = logarithmic_cost(t) * agent_reward_factor[a]
```

当前配置下：

- 默认 agent：`reward_factors.default = -1.0`
- `TerminatorAgent`：`reward_factors.terminator = 0.5`
- Web 类 agent：`reward_factors.web_search = -1.5`

#### 第二步：任务结束后，把每个已执行动作的 reward 再乘真实 cost

代码：

- `puppeteer/inference/policy/REINFORCE_continuous.py:1327-1334`

公式为：

```text
r_step_final = r_base * (action.cost / 100000)
```

这意味着当前数据集里的非终止步 reward，本质上是：

- 与动作步数相关
- 与 agent 类型相关
- 再与实际成本正相关

注意：这不是“任务正确性奖励”，而是带明显人为 shaping 的逐步 reward。

### 3.5 terminal reward 是如何计算的

#### 第一步：任务结束时先得到任务级 reward

在 `GraphReasoning.finalize()` 中：

- `MMLU-Pro`：答对 `+1`，答错 `-1`  
  代码：`puppeteer/inference/reasoning/reasoning.py:201-211`
- `GSM-Hard`：答对 `+1`，答错 `-1`  
  代码：`puppeteer/inference/reasoning/reasoning.py:212-222`
- `SRDD`：`BenchmarkEvaluator.check_srdd()` 返回连续 reward  
  代码：`puppeteer/inference/reasoning/reasoning.py:224-236`
- `CW`：`BenchmarkEvaluator.check_commongen()` 返回连续 reward  
  代码：`puppeteer/inference/reasoning/reasoning.py:237-249`

答案正确性的判断函数在：

- `puppeteer/tasks/evaluator.py:217-262`：`check_mmlu`
- `puppeteer/tasks/evaluator.py:265-287`：`check_gsm8k`

#### 第二步：在 `finalize_task()` 里把任务级 reward 转成终止步 reward

代码：

- `puppeteer/inference/policy/REINFORCE_continuous.py:1336-1372`

对于 `MMLU-Pro` / `GSM-Hard`，终止步 reward 的精确公式为：

```text
若答案正确：
r_terminal = 1 + alpha_term * logarithmic_cost(L)

若答案错误：
r_terminal = -1 - alpha_term * logarithmic_cost(L)
```

其中：

- `alpha_term = agent_reward_factor[TerminatorAgent]`
- `L = len(current_trajectory)`，即终止前已有步数

更一般地写：

```text
r_terminal = r_task + sign(r_task) * alpha_term * logarithmic_cost(L)
```

其中 `r_task` 是 evaluator 返回的任务级 reward。

#### 第三步：如果当前 path 最后一步不是 Terminator，则补一个 synthetic terminator

代码：

- `puppeteer/inference/policy/REINFORCE_continuous.py:1353-1372`

这个 synthetic terminator step 会带：

- `action = TerminatorAgent`
- `reward = r_terminal`
- `selected_confidence = 1.0`
- `state_snapshot = capture_decision_state(global_info)`

### 3.6 return / credit target 是如何得到的

#### Monte Carlo return

代码：

- `_calculate_returns()`：`puppeteer/inference/policy/workflow_dataset_recorder.py:246-252`

公式：

```text
R_t = r_t + gamma * R_{t+1}
```

然后写入：

- `returns.mc_return`
  代码：`puppeteer/inference/policy/workflow_dataset_recorder.py:171-174`

#### h2_return

代码：

- `_discounted_window_return()`：`puppeteer/inference/policy/workflow_dataset_recorder.py:254-267`

公式：

```text
h2_t = r_t + gamma * r_{t+1}
```

如果后面不足 2 步，就截断。

#### credit_targets.leave_one_out_gap

代码：

- 写入位置：`puppeteer/inference/policy/workflow_dataset_recorder.py:175-178`

当前实现：

```text
leave_one_out_gap = returns[step_index]
```

也就是说，它**并不是真正 leave-one-out 反事实 gap**，只是直接复用了 MC return。

### 3.7 next_state_targets（辅助监督）是如何得到的

代码：

- `puppeteer/inference/policy/workflow_dataset_recorder.py:374-413`

当前实现完全是启发式的 dense supervision。

具体公式：

```text
total_steps = max(len(actions), 1)
successful_steps = 成功动作数
failed_steps = 失败动作数
reasoning_successes = 成功的 reasoning 动作数
tool_successes = 成功的非 reasoning 动作数

redundancy = max(成功动作名总数 - 成功动作名去重数, 0) / total_steps
conflict = failed_steps / total_steps
progress = min(successful_steps / max_step_num, 1.0)
coverage = min((reasoning_successes + tool_successes) / total_steps, 1.0)

if done:
    readiness = 1.0
else:
    readiness = min(
        1.0,
        0.35 * I(reasoning_successes > 0)
      + 0.35 * I(tool_successes > 0)
      + 0.30 * (1 - conflict)
    )
```

最后写入：

- `progress_score`
- `coverage_score`
- `conflict_score`
- `redundancy_score`
- `termination_readiness`
- `valid_action_mask`

其中 `valid_action_mask` 目前直接写成：

```text
list(self.agent_role_list)
```

代码：`puppeteer/inference/policy/workflow_dataset_recorder.py:412`

这意味着当前所有 agent 默认都被当作 valid。

---

## 4. JSONL 样本结构与字段说明

### 4.1 顶层结构

单条记录长这样：

```json
{
  "episode_id": "...",
  "path_id": 0,
  "t": 3,
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

写入位置：

- `puppeteer/inference/policy/workflow_dataset_recorder.py:142-187`
- synthetic terminator：`puppeteer/inference/policy/workflow_dataset_recorder.py:201-237`

### 4.2 顶层字段说明表

| 字段 | 来源代码 | 物理含义 | 备注 |
| --- | --- | --- | --- |
| `episode_id` | `workflow_dataset_recorder.py:527-531` | 用 `path_id + workpath + question` 构造的 episode 唯一标识 | 用于 train/val 按 episode 切分 |
| `path_id` | `workflow_dataset_recorder.py:144` | 并行推理路径编号 | 一个问题可对应多个 path |
| `t` | `workflow_dataset_recorder.py:145` | 当前 step 序号 | step-level transition 的时间索引 |
| `task` | `workflow_dataset_recorder.py:146` | 原始任务信息 | 含问题、类型、约束等 |
| `graph` | `workflow_dataset_recorder.py:147` | 动作前图快照 | 是 `s_t` 的一部分 |
| `state` | `workflow_dataset_recorder.py:148` | 动作前 workflow 状态快照 | 是 `s_t` 的另一部分 |
| `next_graph` | `workflow_dataset_recorder.py:149` | 动作后图快照 | 用于构造 `s_{t+1}` |
| `next_state` | `workflow_dataset_recorder.py:150` | 动作后 workflow 状态快照 | 用于构造 `s_{t+1}` |
| `action` | `workflow_dataset_recorder.py:151-159` | 当前执行动作 | 是 `a_t` |
| `next_state_targets` | `workflow_dataset_recorder.py:160-163` | 辅助监督信号 | 启发式构造 |
| `outcome` | `workflow_dataset_recorder.py:164-170` | 当前一步真实结果 | 包含 reward / cost / done |
| `returns` | `workflow_dataset_recorder.py:171-174` | 从该步起的回报 | 主要给 value 头 |
| `credit_targets` | `workflow_dataset_recorder.py:175-178` | 反事实/信用分配监督 | 当前实现近似为 MC return |
| `metadata` | `workflow_dataset_recorder.py:179-186` | 便于分析的文本信息 | 训练时不直接使用 |

### 4.3 `state` 子字段

来自 `_build_state_snapshot()` 和 `_build_workflow_summary()`：

- `puppeteer/inference/policy/workflow_dataset_recorder.py:269-301`
- `puppeteer/inference/policy/workflow_dataset_recorder.py:303-372`

| 子字段 | 含义 | 训练用途 |
| --- | --- | --- |
| `workflow_state` | 把执行过的 `(agent, action, success)` 序列字符串化后的状态标识 | 离散状态 ID |
| `executed_steps` | 历史步骤摘要列表 | step encoder 的输入 |
| `recent_answers` | 最近答案摘要 | evidence 输入的一部分 |
| `reasoning_results` | reasoning 类成功结果摘要 | evidence 输入的一部分 |
| `tool_results` | tool 类成功结果摘要 | evidence 输入的一部分 |
| `all_actions` | 已执行动作名列表 | 目前训练中未直接用 |
| `valid_actions` | 当前 recorder 认为的可选动作列表 | 作为 next_valid_mask 的 fallback |
| `budget.step_index` | 当前第几步 | budget 输入 |
| `budget.used_tokens` | 已用 token | budget 输入 |
| `budget.used_cost` | 已用 cost | budget 输入 |
| `workflow_valid_actions` | 历史成功动作名 | 当前未直接作为监督目标 |
| `path_id` | 当前 path 编号 | 当前未直接用于建模 |

### 4.4 `graph` 子字段

来自 `_build_graph_snapshot()`：

- `puppeteer/inference/policy/workflow_dataset_recorder.py:415-436`

| 子字段 | 含义 | 训练用途 |
| --- | --- | --- |
| `nodes` | agent 角色列表 | graph node ids |
| `edges` | agent 间边结构 | graph adjacency |
| `node_stats.success_rate` | 该 agent 历史成功率 | graph node features |
| `node_stats.avg_cost` | 该 agent 历史平均 cost | graph node features |
| `node_stats.avg_credit` | 该 agent 历史平均 credit | graph node features |
| `node_stats.avg_reward` | 该 agent 历史平均 reward | 当前训练未直接用 |
| `node_stats.usage_count` | 历史调用次数 | graph node features |

### 4.5 `action` 子字段

来自 `record_completed_trajectory()`：

- `puppeteer/inference/policy/workflow_dataset_recorder.py:151-159`

| 子字段 | 含义 | 训练用途 |
| --- | --- | --- |
| `kind` | 当前动作类型，当前基本写死为 `primitive` | `action_kind_ids` |
| `name` | 当前动作名 / agent 名 | `action_name_ids` |
| `selected_confidence` | scheduler 选中概率 | 当前未直接入模 |
| `estimated_cost` | 动作 cost | `action_features` |
| `candidate_agents` | 当前决策时候选 agent 列表 | 当前未直接入模 |

### 4.6 `outcome` 子字段

| 子字段 | 来源 | 含义 |
| --- | --- | --- |
| `reward` | `trajectory_step["reward"]` | 当前 step 的最终 reward target |
| `cost_delta` | `action.cost` | 当前 step 实际 cost |
| `token_delta` | `action.tokens` | 当前 step token 用量 |
| `done` | 是否为终止 step | done 监督 |
| `success` | action.success 归一化后 | 当前未直接作为头监督 |

代码：`puppeteer/inference/policy/workflow_dataset_recorder.py:164-170`

### 4.7 `returns` / `credit_targets`

| 字段 | 含义 | 训练用途 |
| --- | --- | --- |
| `returns.mc_return` | 从当前步开始的 discounted return | `value` target |
| `returns.h2_return` | 2 步截断 return | 当前训练未直接用 |
| `credit_targets.leave_one_out_gap` | 当前实现等于 `mc_return` | `counterfactual_credit` target |
| `credit_targets.step_credit` | 当前 step reward | 当前训练未直接用 |

### 4.8 `next_state_targets`

| 字段 | 含义 | 训练用途 |
| --- | --- | --- |
| `progress_score` | 进度 | aux head |
| `coverage_score` | 信息覆盖度 | aux head |
| `conflict_score` | 冲突/失败程度 | aux head；同时也被当作 uncertainty target |
| `redundancy_score` | 冗余调用程度 | aux head |
| `termination_readiness` | 接近终止程度 | aux head |
| `valid_action_mask` | 下一步有效动作集合 | valid 头 |

---

## 5. `WorkflowStateAdapter` 如何把 JSONL 转成张量

核心代码：

- `puppeteer/inference/policy/workflow_world_model.py:812-1149`

它做了两件事：

1. 扫描离散词表  
   `scan_records()` / `_scan()`  
   代码：`workflow_world_model.py:918-938`

2. 把每条记录映射成定长 batch 张量  
   `build_batch()`  
   代码：`workflow_world_model.py:940-1149`

### 5.1 Batch dataclass

定义位置：

- `WorkflowWorldModelBatch`：`puppeteer/inference/policy/workflow_world_model.py:273-336`
- `WorkflowWorldModelTargets`：`puppeteer/inference/policy/workflow_world_model.py:247-270`

### 5.2 张量总表

| 张量 | 形状 | 来源 |
| --- | --- | --- |
| `task_features` | `[B, task_dim]` | 题面手工文本特征 |
| `task_type_ids` | `[B]` | `task.task_type` / `task.type` |
| `workflow_state_ids` | `[B]` | `state.workflow_state` |
| `step_role_ids` | `[B, max_steps]` | 历史 step agent |
| `step_action_ids` | `[B, max_steps]` | 历史 step action |
| `step_features` | `[B, max_steps, step_dim]` | 历史 step 数值特征 |
| `step_mask` | `[B, max_steps]` | 历史 step 是否存在 |
| `evidence_type_ids` | `[B, max_evidence]` | reasoning/tool/answer 类型 |
| `evidence_features` | `[B, max_evidence, evidence_dim]` | 证据手工特征或占位 |
| `budget_features` | `[B, budget_dim]` | budget 相关特征 |
| `graph_node_ids` | `[B, max_nodes]` | 图节点角色 ID |
| `graph_node_features` | `[B, max_nodes, node_dim]` | 图节点统计特征 |
| `graph_adj` | `[B, max_nodes, max_nodes]` | 图邻接矩阵 |
| `graph_mask` | `[B, max_nodes]` | 节点 mask |
| `action_kind_ids` | `[B]` | 动作类型 ID |
| `action_name_ids` | `[B]` | 动作名 ID |
| `action_features` | `[B, action_dim]` | 当前动作数值特征 |
| `targets.reward` | `[B]` | `outcome.reward` |
| `targets.cost` | `[B]` | 归一化 `outcome.cost_delta` |
| `targets.done` | `[B]` | `outcome.done` |
| `targets.value` | `[B]` | `returns.mc_return` |
| `targets.uncertainty` | `[B]` | `next_state_targets.conflict_score` |
| `targets.next_valid_mask` | `[B, num_actions]` | `valid_action_mask` |
| `targets.aux[name]` | `[B]` | `next_state_targets` 中各 aux |
| `targets.counterfactual_credit` | `[B]` | `credit_targets.leave_one_out_gap` |

### 5.3 题面文本特征

#### 不使用 LLM 编码器时

代码：

- `_text_features()`：`puppeteer/inference/policy/workflow_world_model.py:838-857`
- `build_batch()`：`workflow_world_model.py:1000-1009`

题面文本被压成 8 维统计特征：

1. 字符长度
2. 词数
3. 数字占比
4. 标点占比
5. URL 数
6. 词汇唯一率
7. 是否包含问号
8. 是否包含换行

这是一个非常轻量的 bag-of-statistics 表示。

#### 使用 LLM 编码器时

代码：

- tokenizer：`workflow_world_model.py:859-898`
- 题面 token 化：`workflow_world_model.py:1001-1006`

此时不再使用 `task_features`，而是存入：

- `task_text_input_ids`
- `task_text_attention_mask`

### 5.4 历史步骤特征

代码：

- `workflow_world_model.py:1015-1030`

每个 step 的数值特征为：

1. 是否成功
2. token 数归一化
3. cost 归一化
4. 参数字符串长度
5. answer summary 长度
6. step data summary 长度
7. 是否有 answer summary
8. 是否有 step data summary

这里使用了：

- `_normalize_tokens()`：`workflow_world_model.py:910-912`
- `_normalize_cost()`：`workflow_world_model.py:914-916`

归一化公式都是：

```text
normalized = min(log(1 + raw) / log(1 + clip), 1.0)
```

### 5.5 证据特征

证据由这三类拼起来：

- `reasoning_results`
- `tool_results`
- `recent_answers`

代码：

- `workflow_world_model.py:1032-1047`

并映射到类型：

- reasoning -> 1
- tool -> 2
- answer -> 3

### 5.6 budget 特征

代码：

- `workflow_world_model.py:1049-1060`

4 维特征分别为：

1. 当前 step index
2. 已用 token
3. 已用 cost
4. 任务约束中的 budget 上限

### 5.7 graph 特征

代码：

- `workflow_world_model.py:1062-1083`

每个节点的 6 维特征为：

1. success_rate
2. avg_cost（归一化）
3. avg_credit
4. usage_count
5. 是否 web 类 agent（`TavilyAgent / WebsiteAgent / ArxivAgent`）
6. 是否 `TerminatorAgent`

### 5.8 action 特征

代码：

- `workflow_world_model.py:1085-1098`

当前动作的 6 维特征为：

1. 是否 macro
2. 是否 mutation
3. 名字里是否包含 `Terminator`
4. 是否 web 类 agent
5. action name 长度
6. estimated_cost（归一化）

### 5.9 target 构造

代码：

- `workflow_world_model.py:1100-1113`

重点：

- `reward <- outcome.reward`
- `cost <- normalize(outcome.cost_delta)`
- `done <- outcome.done`
- `value <- returns.mc_return`
- `uncertainty <- next_state_targets.conflict_score`
- `aux <- next_state_targets[name]`
- `counterfactual_credit <- credit_targets.leave_one_out_gap`

请注意：**uncertainty target 和 conflict_score 是同一个来源**。

---

## 6. 模型结构说明

核心文件：

- `puppeteer/inference/policy/workflow_world_model.py`

### 6.1 配置对象

定义：

- `WorkflowWorldModelConfig`
  `puppeteer/inference/policy/workflow_world_model.py:215-244`

关键参数：

- 维度类：`task_dim / step_dim / evidence_dim / budget_dim / node_dim / action_dim`
- 模型类：`embed_dim / model_dim / hidden_dim / latent_dim`
- 结构类：`num_heads / num_layers`
- 截断类：`max_steps / max_evidence / max_nodes`
- 文本类：`use_llm_text_encoder / text_encoder_model_path / text_encoder_freeze / text_encoder_dtype`
- 监督类：`aux_names / loss_weights`

### 6.2 模块总览

#### 1. SequenceEncoder

- 代码：`workflow_world_model.py:373-403`
- 作用：编码历史 step 序列

输入：

- role ids
- action ids
- 数值 step features
- mask

实现：

- role embedding
- action embedding
- numeric projection
- TransformerEncoder
- masked mean pooling

#### 2. SetEncoder

- 代码：`workflow_world_model.py:406-417`
- 作用：编码 evidence set

实现：

- type embedding
- feature projection
- masked mean pooling

#### 3. GraphEncoder

- 代码：`workflow_world_model.py:420-444`
- 作用：编码 agent graph

实现：

- role embedding
- node feature projection
- 邻居聚合（基于归一化邻接矩阵）
- 多层 MLP 更新
- masked mean pooling

#### 4. HFTextEncoder

- 代码：`workflow_world_model.py:447-498`
- 作用：当启用 `--use-llm-text-encoder` 时，对 task/evidence 文本做语义编码

实现要点：

- `AutoModel.from_pretrained(..., trust_remote_code=True, local_files_only=True)`
- 支持冻结 / 非冻结
- 对 `last_hidden_state` 做 masked mean pooling
- 自动推断 hidden size，兼容多种 HF config 字段

### 6.3 主模型 `WorkflowWorldModel`

定义：

- `workflow_world_model.py:501-809`

可以分成 3 个阶段：

#### 阶段 A：编码当前观测，得到 posterior latent

代码：

- `encode_observation()`：`workflow_world_model.py:629-669`

输入分支：

1. task 分支
2. task_type embedding
3. workflow_state embedding
4. 历史 step 分支
5. evidence 分支
6. budget 分支
7. graph 分支
8. 上一步 hidden_state

融合方式：

```text
concat(
  task_repr,
  task_type_repr,
  workflow_repr,
  step_repr,
  evidence_repr,
  budget_repr,
  graph_repr,
  hidden_state
) -> observation_fusion -> fused
```

随后得到：

- `next_hidden = tanh(hidden_adapter([fused, hidden_state]))`
- `post_mean, post_logvar = posterior_head(fused)`
- `latent = sample(post_mean, post_logvar)`（训练时采样，评估时取均值）

#### 阶段 B：结合当前动作做 transition，得到 prior latent

代码：

- `forward()`：`workflow_world_model.py:677-687`

流程：

```text
action_repr = encode_action(...)
transition_input = [latent, action_repr, graph_repr]
prior_hidden = GRUCell(transition_input, hidden_state)
prior_mean, prior_logvar = prior_head([prior_hidden, graph_repr])
```

#### 阶段 C：用 prior 预测各个监督头

代码：

- `workflow_world_model.py:687-706`

头输入：

```text
head_input = [prior_mean, graph_repr, prior_hidden]
```

输出：

- `reward`
- `cost`
- `done_logits`
- `value`
- `uncertainty = softplus(...)`
- `valid_action_logits`
- `aux[name]`

### 6.4 `q_value()` 和 `imagine_rollout()`

#### `q_value()`

代码：

- `workflow_world_model.py:745-746`

定义：

```text
Q ≈ reward - cost + gamma * value
```

这里没有单独的 Q 网络，而是用现有的 `reward / cost / value` 组合成一阶近似。

#### `imagine_rollout()`

代码：

- `workflow_world_model.py:708-743`

作用：

- 在 latent 空间中沿着一串 action embedding 向前 rollout
- 给 planner / reranker 提供 imagined reward / cost / value / return

---

## 7. 训练目标与损失函数

核心代码：

- `compute_losses()`  
  `puppeteer/inference/policy/workflow_world_model.py:748-809`

默认 loss weight：

- `workflow_world_model.py:26-38`

### 7.1 latent transition loss

代码：

- `workflow_world_model.py:760-769`

定义：

```text
latent_loss = SmoothL1(output.prior_mean, next_mean_detached)
```

物理含义：

- 当前步执行动作后预测出的 prior latent
- 要贴近真实 `s_{t+1}` 编码出来的 posterior mean

### 7.2 KL loss

代码：

- KL 函数：`workflow_world_model.py:55-63`
- 使用位置：`workflow_world_model.py:763-769`

定义：

```text
kl = KL( next_posterior || current_predicted_prior )
```

物理含义：

- 约束预测 prior 分布与真实下一状态 posterior 分布一致

### 7.3 reward / cost / value / uncertainty

代码：

- `workflow_world_model.py:775-789`

统一使用：

```text
SmoothL1Loss(pred, target)
```

对应 target 来源：

- `reward <- outcome.reward`
- `cost <- normalize(outcome.cost_delta)`
- `value <- returns.mc_return`
- `uncertainty <- next_state_targets.conflict_score`

### 7.4 done

代码：

- `workflow_world_model.py:781-783`

定义：

```text
BCEWithLogits(done_logits, done_label)
```

### 7.5 valid action

代码：

- `workflow_world_model.py:790-795`

定义：

```text
BCEWithLogits(valid_action_logits, next_valid_mask)
```

这是一个多标签二分类头。

### 7.6 auxiliary losses

代码：

- `workflow_world_model.py:796-800`

定义：

```text
aux_loss = mean_i SmoothL1(output.aux[i], target_aux[i])
```

### 7.7 counterfactual loss

代码：

- `workflow_world_model.py:802-808`
- pairwise ranking：`workflow_world_model.py:66-75`

定义：

```text
q_values = reward - cost + gamma * value
cf_reg = SmoothL1(q_values, counterfactual_targets)
cf_rank = pairwise_ranking_loss(q_values, counterfactual_targets)
counterfactual_loss = cf_reg + cf_rank
```

物理含义：

- 既拟合数值大小
- 又拟合样本间相对排序

### 7.8 total loss

代码：

- `workflow_world_model.py:756-809`

总损失是所有项按权重加和：

```text
total =
  1.00 * latent
+ 0.05 * kl
+ 1.00 * reward
+ 0.50 * cost
+ 0.50 * done
+ 0.50 * value
+ 0.25 * uncertainty
+ 0.25 * valid
+ 0.25 * aux
+ 0.50 * counterfactual
```

---

## 8. 训练脚本、指标与日志系统

核心文件：

- `puppeteer/train_workflow_world_model.py`

### 8.1 训练脚本做了什么

主流程：

- 读取参数：`train_workflow_world_model.py:26-96`
- 搜索 JSONL：`107-122`
- 加载 records：`125-136`
- 按 episode 切 train/val：`164-185`
- 构造 adapter：`209-227`
- 扫词表、冻结词表、建模：`714-730`
- 训练循环：`739-774`
- 按 `val.total` 保存 best checkpoint：`763-773`

### 8.2 `next_batch` 是怎么来的

代码：

- `build_next_state_records()`：`train_workflow_world_model.py:188-196`

它会把每条 record 的：

- `graph <- next_graph`
- `state <- next_state`

从而把同一条样本重写成“下一时刻观测”，供 latent 对齐监督使用。

注意：

- 这里只是为了 `encode_observation(next_batch)` 得到 `next_mean`
- 它并不打算把 `next_batch` 的 target 再用于预测头损失

### 8.3 tracker 上报的指标有哪些

上报函数：

- `log_metrics_to_tracker()`：`train_workflow_world_model.py:386-400`

每个 epoch 都会上报：

- `train/*`
- `val/*`
- `val/best_total`

### 8.4 指标是怎么统计的

#### 回归类指标

代码：

- 累积：`train_workflow_world_model.py:403-419`
- 汇总：`511-524`

输出：

- `*_pred_mean`
- `*_target_mean`
- `*_mae`
- `*_rmse`

适用于：

- `reward`
- `cost`
- `value`
- `uncertainty`
- `aux_*`
- `counterfactual`

#### 二分类指标

代码：

- 累积：`422-439`
- 汇总：`525-530`

输出：

- `done_prob_mean`
- `done_target_mean`
- `done_acc`
- `done_brier`

#### 多标签指标

代码：

- 累积：`442-464`
- 汇总：`531-546`

输出：

- `valid_prob_mean`
- `valid_target_density`
- `valid_label_acc`
- `valid_exact_match`
- `valid_precision`
- `valid_recall`
- `valid_f1`
- `valid_brier`

### 8.5 当前“最佳模型”是按什么选的

代码：

- `train_workflow_world_model.py:759-766`

定义：

```text
best checkpoint = 验证集 total loss 最小的 epoch
```

注意这意味着：

- 最优模型不一定是 rollout 最优
- 不一定是 `value` / `counterfactual` 最优
- 也不一定是 planner 最好用的模型

---

## 9. 当前实现里值得重点注意的问题

这一节最重要。下面是我基于代码阅读得出的当前实现风险清单。

### 9.1 `valid_action_mask` 基本是常量，导致 valid 头和指标几乎无意义

代码：

- `puppeteer/inference/policy/workflow_dataset_recorder.py:406-413`

当前实现：

```text
"valid_action_mask": list(self.agent_role_list)
```

这表示下一步所有 agent 默认都 valid。

后果：

1. `valid` 头学到“全 1”就能得到非常高的 F1。
2. `valid_f1`、`valid_exact_match` 等指标高度乐观。
3. 这个头对 planner/reranker 的约束价值很弱。

这是当前最明确的问题之一。

### 9.2 `uncertainty` target 与 `conflict_score` 完全同源，两个头高度重复

代码：

- target 构造：`puppeteer/inference/policy/workflow_world_model.py:1105-1108`

当前实现：

```text
uncertainty <- next_state_targets["conflict_score"]
aux["conflict_score"] <- next_state_targets["conflict_score"]
```

后果：

1. `uncertainty` 头和 `aux_conflict_score` 头几乎在学同一个东西。
2. 这会制造重复监督，浪费头容量。
3. 指标上看起来像有两个不同能力，实际上不是。

### 9.3 `credit_targets.leave_one_out_gap` 并不是真正 leave-one-out gap

代码：

- `puppeteer/inference/policy/workflow_dataset_recorder.py:175-178`

当前实现直接写：

```text
leave_one_out_gap = returns[step_index]
```

这意味着：

- 当前 counterfactual 目标并不是“拿掉该步后性能下降多少”
- 只是“从该步开始的累计回报”

因此 `counterfactual` 头的名字和实际语义并不一致。

### 9.4 reward 语义比较混杂：既有 correctness，也有强人为 shaping

相关代码：

- 基础 step reward：`REINFORCE_continuous.py:1277-1289`
- 逐步 cost 缩放：`REINFORCE_continuous.py:1327-1334`
- terminal reward：`REINFORCE_continuous.py:1336-1372`
- evaluator correctness：`reasoning.py:201-222`

当前 reward 混合了：

1. agent 类型偏好
2. 步数成本
3. 实际 cost 缩放
4. 终止时任务答对/答错

后果：

- `reward` 头学到的不是“纯任务价值”
- `value` 和 `mc_return` 也会继承这种混合语义
- 解释 `reward_mae` / `value_mae` 时要非常小心

### 9.5 `cost` target 是对数归一化，`reward/value/counterfactual` 却保持原尺度，尺度体系不一致

代码：

- `cost` target：`workflow_world_model.py:1102-1105`
- `_normalize_cost()`：`workflow_world_model.py:914-916`

当前：

- `cost` 是 `log1p` 后再截断到 `[0,1]`
- `reward` 和 `mc_return` 却保留原始 shaped reward 数值

后果：

- 多头损失之间的数值尺度不统一
- 容易出现某些头先学会、某些头长期难学
- 仅通过手工 `loss_weights` 很难完全补偿

### 9.6 `done` 头存在明显的 Terminator shortcut

代码：

- action feature 中显式包含 `1.0 if "Terminator" in action_name else 0.0`
  `workflow_world_model.py:1089-1097`
- label：
  `workflow_world_model.py:1102-1105`

这意味着：

- 如果终止通常伴随 `TerminatorAgent`
- 那么 `done` 头很容易学到“看见 Terminator 就判 done”

这会让 `done_acc` 看起来很好，但不一定表示模型真正理解了何时应该结束。

### 9.7 graph node stats 会跨任务累积，可能引入顺序泄漏和非平稳性

代码：

- recorder 初始化 agent_stats：`workflow_dataset_recorder.py:45-69`
- 更新 stats：`workflow_dataset_recorder.py:438-455`
- 构造 graph snapshot：`workflow_dataset_recorder.py:415-436`
- recorder 在策略构造时只初始化一次：
  `REINFORCE_continuous.py:965-975`

这意味着：

- `agent_stats` 不是每个任务/episode 重置
- 后续任务的 `graph.node_stats` 会携带前面任务的历史统计

后果：

1. 数据分布随采样顺序变化
2. 不同样本之间可能存在跨任务信息泄漏
3. 训练集和验证集如果来自同一次采集流程，这种历史统计会放大“时间顺序偏差”

这是一个很值得认真排查的问题。

### 9.8 当前 aux target 都是启发式，不是外部真值

代码：

- `workflow_dataset_recorder.py:374-413`

这不是 bug，但必须清楚：

- `progress / coverage / conflict / redundancy / readiness`
  都是 recorder 规则算出来的
- 世界模型学到的是“对 heuristic 的拟合”
- 不是“对真实潜在任务状态”的拟合

如果 heuristic 本身不靠谱，模型再强也只是在拟合 heuristic。

### 9.9 LLM 文本编码器方案很重，但当前任务标签并不一定需要强语义

相关代码：

- 训练脚本默认 `task_text_max_length = 8192`
  `train_workflow_world_model.py:65`
- `HFTextEncoder` 使用 mean pooling
  `workflow_world_model.py:482-498`
- evidence/task 都只做 pooled embedding 后线性投影
  `workflow_world_model.py:579-627`

风险：

1. 文本编码很重
2. pooling 很粗
3. 当前数据规模较小时更容易造成 latent/kl 不稳定
4. 但许多监督头本身主要依赖结构化状态，不强依赖深语义

这解释了你目前实验里“不开 LLM 反而更稳”的现象。

### 9.10 `progress_score` 的归一化基准来自在线 `max_step_num`，但模型侧 `max_steps` 是另一套参数

代码：

- 在线 `max_step_num`：`config/global.yaml:17-19`
- recorder 构造 progress：`workflow_dataset_recorder.py:398`
- 模型序列长度：`workflow_world_model.py:230`

当前：

- 在线系统最大步数是 `global.yaml` 里的 `graph.max_step_num`
- 模型 batch padding 长度是 `WorkflowWorldModelConfig.max_steps`

这两个量不一定一致。

后果：

- `progress_score` 的物理含义依赖在线配置
- 但模型接收的历史序列长度上限依赖离线训练配置

### 9.11 现有 README 与代码有轻微漂移

例子：

- `README_Train_world_model.md:31` 提到默认 `dataset-filename` 倾向旧名字
- 实际代码默认是 `workflow_world_model`
  `train_workflow_world_model.py:30-33`

这说明已有文档不能完全替代读代码。

---

## 10. 模型输出、损失、指标的“物理含义”速查表

| 项目 | 代码位置 | 数学/工程定义 | 物理含义 |
| --- | --- | --- | --- |
| `latent` | `workflow_world_model.py:762` | `prior_mean` 对齐 `next_mean` | 下一状态抽象表示是否能被预测 |
| `kl` | `workflow_world_model.py:763-768` | `KL(next posterior || current prior)` | 预测状态分布是否与真实下一状态分布一致 |
| `reward` | `workflow_world_model.py:775-777` | SmoothL1 | 当前 step 的 shaped reward |
| `cost` | `workflow_world_model.py:778-780` | SmoothL1 | 当前 step 的归一化 cost |
| `done` | `workflow_world_model.py:781-783` | BCEWithLogits | 当前 step 是否终止 |
| `value` | `workflow_world_model.py:784-786` | SmoothL1 | 从当前步起的 discounted return |
| `uncertainty` | `workflow_world_model.py:787-789` | SmoothL1 | 当前实现里近似“conflict 程度” |
| `valid` | `workflow_world_model.py:790-795` | 多标签 BCE | 预测下一步哪些动作有效 |
| `aux` | `workflow_world_model.py:796-800` | 多个 SmoothL1 均值 | 拟合启发式 workflow 中间信号 |
| `counterfactual` | `workflow_world_model.py:802-808` | 回归 + 排序 | 近似 Q / credit 信号 |

---

## 11. 如果你要继续扩展/修改，优先建议看哪里

### 11.1 如果你要修数据标签

先看：

1. `puppeteer/inference/reasoning/reasoning.py:193-249`
2. `puppeteer/inference/policy/REINFORCE_continuous.py:1317-1382`
3. `puppeteer/inference/policy/workflow_dataset_recorder.py:85-244`
4. `puppeteer/inference/policy/workflow_dataset_recorder.py:374-413`

### 11.2 如果你要修模型结构

先看：

1. `workflow_world_model.py:501-706`
2. `workflow_world_model.py:748-809`
3. `workflow_world_model.py:940-1149`

### 11.3 如果你要修训练稳定性

先看：

1. `train_workflow_world_model.py:613-670`
2. `workflow_world_model.py:760-809`
3. `train_workflow_world_model.py:511-610`

### 11.4 如果你要修 LLM 文本编码器

先看：

1. `workflow_world_model.py:447-498`
2. `workflow_world_model.py:579-627`
3. `train_workflow_world_model.py:46-71`

---

## 12. 个人建议的后续排查顺序

如果你的目标是“先把这套世界模型做成一个靠谱 baseline”，我建议按下面顺序排：

1. 先修 `valid_action_mask`  
   这是最明显的标签问题。当前 valid 头基本没有真实信息量。

2. 把 `uncertainty` 与 `conflict_score` 解耦  
   否则 uncertainty 头没有独立语义。

3. 明确 `counterfactual_credit` 的真实定义  
   如果它只是 MC return，就不要再叫 leave-one-out gap。

4. 重新审视 reward shaping  
   尤其是“中间步 reward 乘 cost / 100000”和 terminal reward 的符号加减逻辑。

5. 检查 `agent_stats` 是否应按任务或 episode 重置  
   防止 graph node stats 造成顺序泄漏。

6. 在数据量很小时优先使用非 LLM 文本特征 baseline  
   等结构化标签链条更干净、数据更多后，再尝试 LLM 文本编码器。

7. 不要只盯 `val/total`  
   更要盯：
   - `val/latent`
   - `val/kl`
   - `val/value_mae`
   - `val/counterfactual_mae`

---

## 13. 速记版：一句话概括每个核心模块

| 模块 | 一句话概括 |
| --- | --- |
| `GraphReasoning.finalize()` | 任务结束后根据答案正确性或 evaluator 打 terminal reward |
| `ContinuousREINFORCE.append_to_trajectory()` | 每选一步动作，就记录动作前状态和基础 shaping reward |
| `ContinuousREINFORCE.finalize_task()` | 用真实执行结果把整条轨迹的 reward / done / cost 回填完整 |
| `WorkflowDatasetRecorder` | 把在线轨迹压成 JSONL transition 数据集 |
| `WorkflowStateAdapter` | 把 JSONL 转成模型可吃的定长张量 |
| `WorkflowWorldModel.encode_observation()` | 编码当前观测得到 posterior latent |
| `WorkflowWorldModel.forward()` | 结合当前动作预测 prior latent 和各监督头 |
| `WorkflowWorldModel.compute_losses()` | 把 latent / value / reward / aux / counterfactual 等损失汇总 |
| `train_workflow_world_model.py` | 读取数据、训练、评估、记录指标、保存 best checkpoint |

---

## 14. 最后一句结论

当前这套世界模型从工程上已经形成了完整闭环：

```text
在线执行 -> 轨迹回填 -> JSONL -> Batch -> 模型 -> Loss -> 指标
```

但它目前更像一个“结构化 workflow 预测 baseline”，而不是一个已经严谨定义好的世界模型系统。

最核心的现实判断是：

1. 它已经能训练。
2. 它已经有可解释的模块划分。
3. 但它的若干监督目标存在明显近似和捷径。
4. 因此当前实验结果能说明“模型在拟合当前标签体系”，还不能直接说明“模型已经学会了真实可靠的 workflow dynamics”。

如果你后续愿意，我建议下一步直接基于本文第 9 节，把“当前问题清单”拆成一个可执行整改列表，我可以继续帮你写成 `世界模型整改计划.md`。
