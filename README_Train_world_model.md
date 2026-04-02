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
2. 读取 step-level records
3. 划分 train/val
   - 默认模式：按 `episode_id` 切分
   - 手动目录模式：分别从 `--train-data-root` 和 `--test-data-root` 读取
4. 用同一条 record 的 `next_state / next_graph` 构造 `next_batch`
5. `WorkflowStateAdapter.build_batch()` 转成张量
6. `WorkflowWorldModel.compute_losses()` 训练

注意：

- 默认模式下，`episode_id` 不包含 `path_id`
- 因此同题 sibling paths 会被一起划到 train 或 val，不会跨 split 泄漏
- 如果显式传入 `--train-data-root` 和 `--test-data-root`，则不再走 `split_by_episode`

---

## 2. 基础训练

最小用法：

```bash
python train_workflow_world_model.py --data-root results
```

手动指定训练集和测试集目录：

```bash
python train_workflow_world_model.py \
  --train-data-root results/world_model_dataset-llm/MMLU-Pro/train \
  --test-data-root results/world_model_dataset-llm/MMLU-Pro/test \
  --dataset-filename workflow_world_model
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

常用参数：

- `--data-root`
  - 单目录模式下的数据根目录
- `--train-data-root`
  - 手动指定训练集根目录
  - 会递归读取该目录及其子目录中的匹配 JSONL
- `--test-data-root`
  - 手动指定验证/测试集根目录
  - 会递归读取该目录及其子目录中的匹配 JSONL
- `--dataset-filename`
  - 默认值是 `workflow_world_model`
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
  - 默认模式下的验证集比例
- `--use-llm-text-encoder`
  - 启用 HuggingFace 文本编码器

### 3.1 `--dataset-filename` 的含义

当前脚本默认值：

```text
workflow_world_model
```

搜索逻辑：

- 如果参数本身不带 `.jsonl`
  - 按文件名前缀匹配
- 如果参数本身带 `.jsonl`
  - 同时兼容精确文件名和同名前缀文件

这正是为了兼容 recorder 生成的时间戳文件名。

### 3.2 `--train-data-root` / `--test-data-root` 的行为

如果这两个参数同时提供，则脚本会进入“手动目录切分模式”：

- `train_records` 只从 `--train-data-root` 递归读取
- `val_records` 只从 `--test-data-root` 递归读取
- 不再执行按 `episode_id` 的随机切分
- `--val-ratio` 在该模式下不再生效

如果只提供其中一个参数，脚本会直接报错。

---

## 4. 当前 target 的真实语义

当前 world model 的监督目标已经不是旧版“原始逐步 reward”。

当前实际映射是：

- `reward <- outcome.reward`
  - 当前主语义是树回传的 `action_value`
- `cost <- normalize(outcome.cost_delta)`
  - 当前 step 的真实成本
- `done <- outcome.done`
- `value <- returns.mc_return`
  - 当前主语义通常也是树回传值
- `uncertainty <- next_state_targets.conflict_score`
- `counterfactual_credit <- credit_targets.leave_one_out_gap`
  - 当前主语义是相对 sibling baseline 的优势

因此训练时可以这样理解：

- `reward` 头学的是过程价值
- `value` 头学的是当前边对应的树回传值
- `counterfactual` 头学的是相对兄弟分支的边际优势

---

## 5. `next_batch` 是怎么来的

训练脚本不会跨记录查找“下一条样本”，而是直接把同一条 record 重写成“下一时刻观测”：

```text
graph <- next_graph
state <- next_state
```

这要求 recorder 数据本身满足：

- `graph` 是 pre-action
- `next_graph` 是 post-action
- `state` 是 pre-action
- `next_state` 是 post-action

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

### 6.2 手动指定 train/test 目录

```bash
python train_workflow_world_model.py \
  --train-data-root results/world_model_dataset-llm/MMLU-Pro/train \
  --test-data-root results/world_model_dataset-llm/MMLU-Pro/test \
  --dataset-filename workflow_world_model \
  --epochs 10 \
  --batch-size 32 \
  --device cuda
```

### 6.3 启用 LLM 文本编码器

```bash
python train_workflow_world_model.py \
  --data-root results/world_model_dataset-llm \
  --use-llm-text-encoder \
  --text-encoder-model-path /extrahome0/HF_models/Qwen/Qwen3.5-4B \
  --device cuda
```

### 6.4 小规模调试

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

- 总 loss 和各分项 loss
- `reward / cost / value / uncertainty / counterfactual` 的 `mae / rmse`
- `done` 的 `acc / brier`
- `valid_action` 的 `f1 / precision / recall / label_acc / exact_match`
- `aux_*` 的误差

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

1. 如果 `reward_mae` 下降
   - 说明模型更会拟合树回传的过程价值
2. 如果 `counterfactual_mae` 下降
   - 说明模型更会区分“当前分支相对兄弟分支更好还是更差”
3. 如果 `valid_f1` 很高
   - 仍要谨慎，因为当前 `valid_action_mask` 还是常量式标签
4. 如果 `done_acc` 很高
   - 仍要谨慎，因为 action feature 中可能存在 `Terminator` shortcut

---

## 11. 推荐排查顺序

如果训练效果不理想，建议按这个顺序排查：

1. 先抽查 JSONL
   - 看 `outcome.reward / returns / credit_targets` 是否真的是树回传值
2. 再看 `episode_id`
   - 确认同题 sibling paths 是否共享同一 episode
3. 再看 `valid_action_mask`
   - 确认你是否接受当前常量式标签
4. 最后再调模型结构和文本编码器
