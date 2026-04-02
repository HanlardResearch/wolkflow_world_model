# World Model 指标说明

本文档说明 [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) 在 `W&B` 和 `SwanLab` 中记录的指标含义。

两种平台记录的是同一套指标名。

- `train/*` 表示训练集指标
- `val/*` 表示验证集指标
- `val/best_total` 表示截至当前 epoch 的最佳验证集总损失

## 世界模型架构图

![世界模型架构图](puppeteer/figs/世界模型架构-LLM.png)

## 1. 指标命名规则

训练脚本会按 epoch 上报以下几类指标：

- loss 指标
- 回归头误差指标
- 二分类指标
- 多标签动作预测指标
- 辅助目标指标
- 反事实 credit 头指标

常见命名形式如下：

- `train/reward`
- `val/kl`
- `train/reward_mae`
- `val/done_acc`
- `train/valid_f1`
- `val/aux_progress_score_mae`
- `train/counterfactual_rmse`

### 1.1 指标定义位置

指标的来源分成两层：

- loss 定义：
  - 在 [inference/policy/workflow_world_model.py](puppeteer/inference/policy/workflow_world_model.py) line 675 到 [workflow_world_model.py](puppeteer/inference/policy/workflow_world_model.py) line 725
- 评估指标累计与聚合：
  - 在 [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 403 到 [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 546
- batch 级指标汇总入口：
  - 在 [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 467 到 [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 507
- epoch 级指标生成：
  - 在 [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 511 到 [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 546
- tracker 上报：
  - 在 [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 394 到 [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 400 
## 2. 总损失与分项损失

这些指标直接来自模型训练时的 loss 计算。

### `train/total` / `val/total`

总损失。

它是各分项损失按权重加权后的和，是训练时真正反向传播的目标。  
如果这个值下降，说明整体优化目标在改善；如果它剧烈波动，通常需要先看是哪一项子损失在主导波动。

定义位置：

- 总损失初始化和累加：
  - [inference/policy/workflow_world_model.py](puppeteer/inference/policy/workflow_world_model.py) line 670
- [inference/policy/workflow_world_model.py](puppeteer/inference/policy/workflow_world_model.py) line 684
- [inference/policy/workflow_world_model.py](puppeteer/inference/policy/workflow_world_model.py) line 692
- [inference/policy/workflow_world_model.py](puppeteer/inference/policy/workflow_world_model.py) line 725 
### `train/latent` / `val/latent`

latent 对齐损失。

含义是模型预测的 `prior_mean` 与真实下一状态编码得到的 `next posterior mean` 之间的 `Smooth L1 Loss`。  
它衡量 transition 是否能把当前状态和动作正确映射到下一时刻 latent。

定义位置：

- [inference/policy/workflow_world_model.py](puppeteer/inference/policy/workflow_world_model.py) line 677 
### `train/kl` / `val/kl`

KL 散度损失。

含义是下一状态 posterior 分布相对于当前模型预测 prior 分布的 KL。  
它约束 latent 分布不要漂移过大，帮助 prior/posterior 对齐。

定义位置：

- KL 公式：
  - [inference/policy/workflow_world_model.py](puppeteer/inference/policy/workflow_world_model.py) line 61
- KL 损失使用位置：
  - [inference/policy/workflow_world_model.py](puppeteer/inference/policy/workflow_world_model.py) line 678 
### `train/reward` / `val/reward`

reward 头的监督损失。

模型预测的是下一步 reward。  
损失函数是 `Smooth L1 Loss`。

定义位置：

- loss：
  - [inference/policy/workflow_world_model.py](puppeteer/inference/policy/workflow_world_model.py) line 691
- 评估统计：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 483 
### `train/cost` / `val/cost`

cost 头的监督损失。

模型预测的是下一步 cost。  
这里的 target 是 recorder 数据中的 `cost_delta` 经归一化后的值。  
损失函数是 `Smooth L1 Loss`。

定义位置：

- loss：
  - [inference/policy/workflow_world_model.py](puppeteer/inference/policy/workflow_world_model.py) line 694
- 评估统计：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 485 
### `train/done` / `val/done`

done 头的监督损失。

模型预测的是 episode 是否结束。  
损失函数是 `BCEWithLogitsLoss`。

定义位置：

- loss：
  - [inference/policy/workflow_world_model.py](puppeteer/inference/policy/workflow_world_model.py) line 697
- 评估统计：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 491 
### `train/value` / `val/value`

value 头的监督损失。

模型预测的是一步状态价值，当前实现默认用数据中的 `mc_return` 或 reward 近似监督。  
损失函数是 `Smooth L1 Loss`。

定义位置：

- loss：
  - [inference/policy/workflow_world_model.py](puppeteer/inference/policy/workflow_world_model.py) line 700
- 评估统计：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 487 
### `train/uncertainty` / `val/uncertainty`

uncertainty 头的监督损失。

当前默认使用 `next_state_targets.conflict_score` 作为 uncertainty target。  
损失函数是 `Smooth L1 Loss`。

定义位置：

- loss：
  - [inference/policy/workflow_world_model.py](puppeteer/inference/policy/workflow_world_model.py) line 703
- 评估统计：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 489 
### `train/valid` / `val/valid`

valid action 多标签损失。

模型预测下一时刻哪些动作是有效动作。  
损失函数是 `BCEWithLogitsLoss`，目标来自 `next_state_targets.valid_action_mask`。

定义位置：

- loss：
  - [inference/policy/workflow_world_model.py](puppeteer/inference/policy/workflow_world_model.py) line 707
- 评估统计：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 493 
### `train/aux` / `val/aux`

辅助目标总损失。

它是多个辅助分数头的平均 `Smooth L1 Loss`，包括：

- `progress_score`
- `coverage_score`
- `conflict_score`
- `redundancy_score`
- `termination_readiness`

定义位置：

- loss：
  - [inference/policy/workflow_world_model.py](puppeteer/inference/policy/workflow_world_model.py) line 713
- 评估统计：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 501 
### `train/counterfactual` / `val/counterfactual`

反事实 credit 头损失。

它由两部分组成：

- 数值拟合项 `cf_reg`
- 排序拟合项 `cf_rank`

目标通常来自 `credit_targets.leave_one_out_gap`。

定义位置：

- Q 值定义：
  - [inference/policy/workflow_world_model.py](puppeteer/inference/policy/workflow_world_model.py) line 657
- 排序损失公式：
  - [inference/policy/workflow_world_model.py](puppeteer/inference/policy/workflow_world_model.py) line 73
- counterfactual loss：
  - [inference/policy/workflow_world_model.py](puppeteer/inference/policy/workflow_world_model.py) line 719
- 评估统计：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 502 
## 3. 回归头误差指标

以下指标适用于这些预测头：

- `reward`
- `cost`
- `value`
- `uncertainty`
- `counterfactual`

每个头都会记录 4 个辅助指标。

### `*_mae`

平均绝对误差。

例如：

- `train/reward_mae`
- `val/value_mae`

值越小越好。  
它反映预测值与目标值的平均绝对偏差，直观性最好。

定义位置：

- MAE 聚合：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 523 
### `*_rmse`

均方根误差。

例如：

- `train/reward_rmse`
- `val/counterfactual_rmse`

值越小越好。  
它对大误差更敏感，如果 RMSE 远高于 MAE，通常说明存在少量偏差很大的样本。

定义位置：

- RMSE 聚合：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 524 
### `*_pred_mean`

预测值平均值。

例如：

- `train/reward_pred_mean`
- `val/cost_pred_mean`

这个指标用于观察模型输出的整体偏置。  
如果 `pred_mean` 长期明显高于或低于 `target_mean`，说明模型存在系统性高估或低估。

定义位置：

- `pred_mean` 聚合：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 521 
### `*_target_mean`

目标值平均值。

例如：

- `train/reward_target_mean`
- `val/uncertainty_target_mean`

它表示当前数据切片上标签本身的平均水平。  
配合 `pred_mean` 使用，用来判断模型输出是否偏离数据分布。

定义位置：

- `target_mean` 聚合：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 522 
## 4. Done 二分类指标

这些指标对应 `done` 头。

### `train/done_acc` / `val/done_acc`

done 分类准确率。

表示模型对“是否终止”的离散判断有多少比例是正确的。

定义位置：

- 二分类指标累计：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 422
- `done_acc` 聚合：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 529 
### `train/done_brier` / `val/done_brier`

done 的 Brier Score。

它衡量预测概率与真实标签之间的均方误差。  
值越小越好。  
相比 `acc`，它更能反映概率校准质量。

定义位置：

- Brier 聚合：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 530 
### `train/done_prob_mean` / `val/done_prob_mean`

done 预测概率均值。

它表示模型平均有多大概率认为 episode 会结束。

定义位置：

- `prob_mean` 聚合：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 528 
### `train/done_target_mean` / `val/done_target_mean`

done 标签均值。

它表示当前数据中真实终止样本的比例。  
如果 `prob_mean` 长期明显高于或低于 `target_mean`，说明终止概率预测存在偏差。

定义位置：

- `target_mean` 聚合：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 528 
## 5. Valid Action 多标签指标

这些指标对应 `valid_action` 预测头。

### `train/valid_f1` / `val/valid_f1`

多标签 F1。

综合考虑 precision 和 recall。  
如果你想看“模型是否整体上能把有效动作集合预测对”，这是最重要的指标之一。

定义位置：

- 多标签指标累计：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 442
- F1 聚合：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 543
- [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 545 
### `train/valid_precision` / `val/valid_precision`

多标签精确率。

表示模型预测为有效的动作里，有多少是真的有效动作。  
高 precision 说明模型更保守，误报更少。

定义位置：

- precision 聚合：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 543 
### `train/valid_recall` / `val/valid_recall`

多标签召回率。

表示真实有效动作里，有多少被模型找到了。  
高 recall 说明模型不容易漏掉可选动作。

定义位置：

- recall 聚合：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 544 
### `train/valid_label_acc` / `val/valid_label_acc`

标签级准确率。

它按动作维度逐位计算正确率，再对所有动作标签求平均。  
因为负样本通常更多，这个值往往会高于 F1，需要结合 F1 一起看。

定义位置：

- `label_acc` 聚合：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 541 
### `train/valid_exact_match` / `val/valid_exact_match`

样本级完全匹配率。

表示一整条样本上的有效动作集合是否被完整预测正确。  
这是最严格的动作集合指标，通常会比 `label_acc` 低很多。

定义位置：

- `exact_match` 聚合：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 542 
### `train/valid_brier` / `val/valid_brier`

多标签概率的 Brier Score。

表示每个动作标签上的预测概率与真实标签之间的均方误差。  
值越小越好，用于评估概率输出是否校准。

定义位置：

- 多标签 Brier 聚合：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 546 
## 6. 辅助目标指标

这些指标以 `aux_` 开头。

当前默认包括：

- `aux_progress_score_mae`
- `aux_coverage_score_mae`
- `aux_conflict_score_mae`
- `aux_redundancy_score_mae`
- `aux_termination_readiness_mae`

### `train/aux_progress_score_mae` / `val/aux_progress_score_mae`

进展分数预测误差。

表示模型对“任务推进程度”的估计误差。

定义位置：

- 辅助头 loss：
  - [inference/policy/workflow_world_model.py](puppeteer/inference/policy/workflow_world_model.py) line 713
- `aux_progress_score_mae` 统计入口：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 501 
### `train/aux_coverage_score_mae` / `val/aux_coverage_score_mae`

覆盖度分数预测误差。

表示模型对“当前证据或解答覆盖问题程度”的估计误差。

### `train/aux_conflict_score_mae` / `val/aux_conflict_score_mae`

冲突分数预测误差。

表示模型对“当前信息冲突程度”的估计误差。

### `train/aux_redundancy_score_mae` / `val/aux_redundancy_score_mae`

冗余分数预测误差。

表示模型对“当前执行和信息是否重复冗余”的估计误差。

### `train/aux_termination_readiness_mae` / `val/aux_termination_readiness_mae`

终止准备度误差。

表示模型对“是否已经接近可终止状态”的估计误差。

这些辅助指标本身不是最终目标，但会影响世界模型中间表征的质量。  
如果这些指标下降，通常说明 latent 对 workflow 结构信息的建模更细。

## 7. Counterfactual 指标

这些指标对应 world model 的 `q_head` 或反事实 credit 头。

### `train/counterfactual_mae` / `val/counterfactual_mae`

反事实 credit 预测的平均绝对误差。

定义位置：

- 统计入口：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 502
- MAE 聚合：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 523 
### `train/counterfactual_rmse` / `val/counterfactual_rmse`

反事实 credit 预测的均方根误差。

定义位置：

- RMSE 聚合：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 524 
### `train/counterfactual_pred_mean` / `val/counterfactual_pred_mean`

模型预测的反事实 credit 平均值。

定义位置：

- `pred_mean` 聚合：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 521 
### `train/counterfactual_target_mean` / `val/counterfactual_target_mean`

真实反事实 credit 的平均值。

定义位置：

- `target_mean` 聚合：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 522 
这些指标用于观察模型是否能学到“某个动作或 agent 对结果的边际贡献”。

## 8. `val/best_total`

截至当前 epoch 的最佳验证集总损失。

它不是当前 epoch 的瞬时值，而是历史最优值。  
如果当前 `val/total` 没有下降，`val/best_total` 会保持不变。

定义位置：

- best 值写入 tracker：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 399
- best 值更新逻辑：
  - [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) line 763 
这个指标通常用于：

- 观察训练是否还在刷新最优结果
- 决定是否早停
- 对齐最佳 checkpoint 的保存时机

## 9. 如何解读这些曲线

### 如果 `total` 波动很大

优先看：

- `kl`
- `latent`
- `counterfactual`

因为这些项通常最容易在中后期主导总损失波动。

### 如果 `reward/cost/value` loss 下降，但 `*_mae` 不降

通常说明：

- 训练目标下降幅度不够直观
- 数据尺度不均匀
- 存在少量大误差样本

这时应重点看：

- `*_rmse`
- `*_pred_mean`
- `*_target_mean`

### 如果 `valid_label_acc` 很高，但 `valid_f1` 很低

通常说明负标签太多，模型主要是在预测“无效”。  
此时 `label_acc` 参考价值有限，应优先看：

- `valid_f1`
- `valid_precision`
- `valid_recall`
- `valid_exact_match`

### 如果 `done_acc` 很高，但 `done_brier` 不好

说明离散分类对了很多，但概率校准差。  
模型可能只是在输出过于极端或偏置的结束概率。

## 10. 平台说明

本 README 虽然命名为 `README_wandb.md`，但内容同样适用于 `SwanLab`。

原因是训练脚本内部对两者使用了统一的指标 payload，差别只在记录后端，不在指标定义。
