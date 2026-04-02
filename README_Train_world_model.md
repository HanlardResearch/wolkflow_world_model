# 训练用法 README

训练脚本：

- [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py)

建议先进入 `puppeteer` 目录：

```bash
cd puppeteer
```

---

## 1. 当前训练链路

训练脚本当前的处理流程是：

1. 递归查找 JSONL 文件  
   代码：`train_workflow_world_model.py:107`
2. 读取 step-level records  
   代码：`train_workflow_world_model.py:125`
3. 按 `episode_id` 切 train/val  
   代码：`train_workflow_world_model.py:164`
4. 用同一条 record 的 `next_state / next_graph` 构造 `next_batch`  
   代码：`train_workflow_world_model.py:188`
5. `WorkflowStateAdapter.build_batch()` 转成张量  
   代码：`inference/policy/workflow_world_model.py:982`
6. `WorkflowWorldModel.compute_losses()` 训练  
   代码：`inference/policy/workflow_world_model.py:780`

注意：

- 现在 `episode_id` 已经不包含 `path_id`
- 因此同题 sibling paths 会被一起划到 train 或 val，不会跨 split 泄漏

---

## 2. 基础训练

最小用法：

```bash
python train_workflow_world_model.py --data-root results
```

按单个数据集目录训练：

```bash
python train_workflow_world_model.py \
  --data-root results/world_model_dataset-llm/MMLU-Pro/validation \
  --max-records 500 \
  --epochs 3 \
  --batch-size 8 \
  --device cuda
```

---

## 3. 关键参数

参数定义见：

- `train_workflow_world_model.py:26`

当前常用参数：

- `--data-root`
  - 数据根目录
- `--dataset-filename`
  - 默认值是 `workflow_world_model`
  - 不是旧文档中的 `workflow_dataset.jsonl`
  - 既支持精确文件名，也支持按前缀匹配带时间戳的文件
- `--max-files`
  - 最多读取多少个文件
- `--max-records`
  - 最多读取多少条记录
- `--output-dir`
  - checkpoint 输出目录
- `--epochs`
  - 训练轮数
- `--batch-size`
  - batch 大小
- `--device`
  - `cpu` 或 `cuda`
- `--learning-rate`
  - 学习率
- `--weight-decay`
  - 权重衰减
- `--val-ratio`
  - 验证集比例
- `--use-llm-text-encoder`
  - 启用 HF 文本编码器

### 3.1 `--dataset-filename` 的当前含义

当前脚本默认值：

```text
workflow_world_model
```

对应：

- `train_workflow_world_model.py:30-33`

搜索逻辑是：

- 如果参数本身不带 `.jsonl`
  - 就按前缀匹配
- 如果参数带 `.jsonl`
  - 就兼容精确文件名和同名前缀文件

对应：

- `train_workflow_world_model.py:107`

这正是为了兼容 recorder 的时间戳文件名。

---

## 4. 当前 target 的真实语义

当前 world model 的监督目标已经不是旧版“原始逐步负 reward”。

Adapter 中的实际映射见：

- `inference/policy/workflow_world_model.py:1151-1160`

具体是：

- `reward <- outcome.reward`
  - 当前主语义是树回传的 `action_value`
- `cost <- normalize(outcome.cost_delta)`
  - 仍是当前 step 的真实成本
- `done <- outcome.done`
- `value <- returns.mc_return`
  - 当前主语义通常与 `action_value` 同量纲
- `uncertainty <- next_state_targets.conflict_score`
- `counterfactual_credit <- credit_targets.leave_one_out_gap`
  - 当前主语义是相对 sibling baseline 的优势

因此训练时要这样理解：

- `reward` 头学的是过程价值
- `value` 头学的是当前边对应的树回传值
- `counterfactual` 头学的是边际优势，而不是旧版的 MC return

---

## 5. `next_batch` 是怎么来的

训练脚本不会跨记录查找下一条样本，而是直接把同一条 record 重写成“下一时刻观测”：

```text
graph <- next_graph
state <- next_state
```

对应：

- `train_workflow_world_model.py:188`

这要求 recorder 数据本身必须满足：

- `graph` 是 pre-action
- `next_graph` 是 post-action
- `state` 是 pre-action
- `next_state` 是 post-action

当前代码已经按这个语义修正完毕。

---

## 6. 常见训练命令

### 6.1 普通结构化特征训练

```bash
python train_workflow_world_model.py \
  --data-root results/world_model_dataset-llm \
  --dataset-filename workflow_world_model \
  --epochs 10 \
  --batch-size 32 \
  --device cuda
```

### 6.2 启用 LLM 文本编码器

```bash
python train_workflow_world_model.py \
  --data-root results/world_model_dataset-llm \
  --use-llm-text-encoder \
  --text-encoder-model-path /extrahome0/HF_models/Qwen/Qwen3.5-4B \
  --device cuda
```

### 6.3 小规模调试

```bash
python train_workflow_world_model.py \
  --data-root results/world_model_dataset-llm/MMLU-Pro/validation \
  --max-files 2 \
  --max-records 200 \
  --epochs 2 \
  --batch-size 4 \
  --device cpu
```

---

## 7. 指标输出会看到什么

训练过程中会输出：

- 总损失和各分项损失
- `reward / cost / value / uncertainty / counterfactual` 的 `mae / rmse`
- `done` 的 `acc / brier`
- `valid_action` 的 `f1 / precision / recall / label_acc / exact_match`
- `aux_*` 的误差

指标记录函数在：

- `train_workflow_world_model.py:386`
- `train_workflow_world_model.py:467`
- `train_workflow_world_model.py:511`

---

## 8. 使用 W&B

```bash
python train_workflow_world_model.py \
  --data-root results/world_model_dataset-llm/MMLU-Pro/validation \
  --use-wandb \
  --wandb-project workflow-world-model \
  --wandb-run-name mmlu-pro-validation
```

可选参数：

- `--wandb-entity`
- `--wandb-tags`
- `--wandb-mode`

---

## 9. 使用 SwanLab

```bash
python train_workflow_world_model.py \
  --data-root results/world_model_dataset-llm/MMLU-Pro/validation \
  --use-swanlab \
  --swanlab-project workflow-world-model \
  --swanlab-run-name mmlu-pro-validation
```

可选参数：

- `--swanlab-workspace`
- `--swanlab-tags`
- `--swanlab-mode`

---

## 10. 当前训练结果的解释建议

1. 若 `reward_mae` 下降
   - 说明模型更会拟合树回传的过程价值
2. 若 `counterfactual_mae` 下降
   - 说明模型更会区分“当前分支比兄弟分支更好还是更差”
3. 若 `valid_f1` 很高
   - 仍要谨慎，因为当前 `valid_action_mask` 还是常量式标签
4. 若 `done_acc` 很高
   - 仍要谨慎，因为 action feature 中存在 Terminator shortcut

---

## 11. 推荐排查顺序

如果训练效果不理想，建议按这个顺序排查：

1. 先抽查 JSONL  
   看 `outcome.reward / returns / credit_targets` 是否真是树回传值
2. 再看 `episode_id`  
   确认同题 sibling paths 是否共享同一 episode
3. 再看 `valid_action_mask`
   确认你是否接受当前常量式标签
4. 最后再调模型结构和文本编码器
