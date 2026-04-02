# 训练用法 README


训练脚本：
[train_workflow_world_model.py](puppeteer/train_workflow_world_model.py)

建议先进入 `puppeteer` 目录：

```bash
cd puppeteer
```

### 14.1 基础训练

最小用法：

```bash
python train_workflow_world_model.py --data-root results/world_model_dataset
```

按单个数据集目录训练：

```bash
python train_workflow_world_model.py \
  --data-root results/world_model_dataset/MMLU-Pro/validation \
  --max-records 500 \
  --epochs 3 \
  --batch-size 8 \
  --device cuda
```

常用参数：

- `--data-root`：数据根目录，脚本会递归查找 JSONL 数据文件
- `--dataset-filename`：限定要读取的文件名，默认是 `workflow_dataset.jsonl`
- `--max-files`：最多读取多少个数据文件
- `--max-records`：最多读取多少条记录
- `--epochs`：训练轮数
- `--batch-size`：批大小
- `--device`：训练设备，例如 `cpu`、`cuda`
- `--output-dir`：模型输出目录
- `--learning-rate`：优化器学习率
- `--seed`：随机种子

训练过程中会在终端输出：

- 总损失和各分项损失
- `reward/cost/value/uncertainty/counterfactual` 的 `mae/rmse`
- `done` 的 `acc/brier`
- `valid_action` 的 `f1/precision/recall/label_acc/exact_match`
- 辅助目标 `progress/coverage/conflict/redundancy/readiness` 的误差

### 14.2 使用 W&B 记录指标

先在外部配置 API Key：

```bash
export WANDB_API_KEY="your_api_key"
```

然后运行：

```bash
python train_workflow_world_model.py \
  --data-root results/world_model_dataset/MMLU-Pro/validation \
  --max-records 500 \
  --epochs 3 \
  --batch-size 8 \
  --device cuda \
  --use-wandb \
  --wandb-project workflow-world-model \
  --wandb-run-name mmlu-pro-validation
```

可选参数：

- `--wandb-entity`：团队或个人实体名
- `--wandb-tags`：逗号分隔的标签
- `--wandb-mode`：`online`、`offline` 或 `disabled`

### 14.3 使用 SwanLab 记录指标

先在外部完成 SwanLab 登录或配置对应凭证，然后运行：

```bash
python train_workflow_world_model.py \
  --data-root results/world_model_dataset/MMLU-Pro/validation \
  --max-records 500 \
  --epochs 3 \
  --batch-size 8 \
  --device cuda \
  --use-swanlab \
  --swanlab-project workflow-world-model \
  --swanlab-run-name mmlu-pro-validation \
  --use-llm-text-encoder
```

可选参数：

- `--swanlab-workspace`：SwanLab 工作空间
- `--swanlab-tags`：逗号分隔的标签
- `--swanlab-mode`：`online`、`offline` 或 `disabled`

### 14.4 记录器选择规则

- `--use-wandb` 和 `--use-swanlab` 只能二选一，不能同时开启
- 两个参数都不传时，只在终端打印训练指标
- 使用 W&B 或 SwanLab 之前，需要先在环境中安装对应 SDK
- 训练脚本会按 epoch 上报 `train/*` 和 `val/*` 指标，便于平台侧可视化
