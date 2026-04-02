# World Model 指标说明

本文档说明 [train_workflow_world_model.py](puppeteer/train_workflow_world_model.py) 在 W&B 和 SwanLab 中记录的指标含义。

两种平台使用的是同一套指标名：

- `train/*`
- `val/*`
- `val/best_total`

当前版本最重要的变化是：

- `reward` 头的 target 不再主要表示旧版逐步负成本 reward，而是树回传得到的 `action_value`
- `counterfactual` 头的 target 不再等于 MC return，而是 `leave_one_out_gap`
- `value` 头当前通常与树回传值处于同一量纲

对应代码：

- target 映射：`puppeteer/inference/policy/workflow_world_model.py:1151-1160`
- loss 定义：`puppeteer/inference/policy/workflow_world_model.py:780`
- tracker 上报：`puppeteer/train_workflow_world_model.py:386`
- batch 指标累计：`puppeteer/train_workflow_world_model.py:467`
- epoch 指标汇总：`puppeteer/train_workflow_world_model.py:511`

---

## 1. 指标命名规则

常见命名形式：

- `train/reward`
- `val/kl`
- `train/reward_mae`
- `val/done_acc`
- `train/valid_f1`
- `val/aux_progress_score_mae`
- `train/counterfactual_rmse`

---

## 2. 总损失与分项损失

### `train/total` / `val/total`

总损失。

它是所有分项损失按权重加和后的值，是训练时真正反向传播的目标。

### `train/latent` / `val/latent`

latent 对齐损失。

含义：

- 当前步执行动作后预测出的 `prior_mean`
- 与真实下一状态编码得到的 `next_mean`
- 做 `SmoothL1Loss`

对应：

- `workflow_world_model.py:792-800`

### `train/kl` / `val/kl`

KL 散度损失。

含义：

- 约束当前预测 prior 分布与真实下一状态 posterior 分布一致

对应：

- `workflow_world_model.py:793-800`

### `train/reward` / `val/reward`

reward 头的监督损失。

当前 target 来源：

- `outcome.reward`

当前主语义：

- 树回传得到的 `action_value`
- 也就是 `Q(prefix, action)`

loss 位置：

- `workflow_world_model.py:810`

### `train/cost` / `val/cost`

cost 头的监督损失。

当前 target 来源：

- `normalize(outcome.cost_delta)`

loss 位置：

- `workflow_world_model.py:813`

### `train/done` / `val/done`

done 头的监督损失。

当前 target 来源：

- `outcome.done`

loss 位置：

- `workflow_world_model.py:816`

### `train/value` / `val/value`

value 头的监督损失。

当前 target 来源：

- `returns.mc_return`

在当前 recorder 主链路中，它通常与 `action_value` 同量纲，因为：

- `returns.mc_return` 优先使用 `world_model_mc_return`
- 当前树回传实现里 `world_model_mc_return = action_value`

loss 位置：

- `workflow_world_model.py:819`

### `train/uncertainty` / `val/uncertainty`

uncertainty 头的监督损失。

当前 target 来源：

- `next_state_targets.conflict_score`

loss 位置：

- `workflow_world_model.py:822`

### `train/valid` / `val/valid`

valid action 多标签损失。

当前 target 来源：

- `next_state_targets.valid_action_mask`

loss 位置：

- `workflow_world_model.py:826`

注意：

- 当前 `valid_action_mask` 仍是常量式标签
- 所以这组指标要谨慎解释

### `train/aux` / `val/aux`

辅助目标总损失。

当前默认包含：

- `progress_score`
- `coverage_score`
- `conflict_score`
- `redundancy_score`
- `termination_readiness`

loss 位置：

- `workflow_world_model.py:834`

### `train/counterfactual` / `val/counterfactual`

反事实信用分配损失。

当前 target 来源：

- `credit_targets.leave_one_out_gap`

当前主语义：

- 当前边相对兄弟分支基线的优势

loss 位置：

- `workflow_world_model.py:843`

---

## 3. 回归类误差指标

以下头会记录：

- `*_mae`
- `*_rmse`
- `*_pred_mean`
- `*_target_mean`

适用对象：

- `reward`
- `cost`
- `value`
- `uncertainty`
- `counterfactual`

### `*_mae`

平均绝对误差。

解释时要注意：

- `reward_mae`
  - 不是“原始策略 reward 的误差”
  - 而是树回传 `action_value` 的误差
- `counterfactual_mae`
  - 不是旧版 MC return 的误差
  - 而是 sibling-baseline advantage 的误差

### `*_rmse`

均方根误差。

如果 RMSE 明显高于 MAE，通常说明存在少量偏差很大的样本。

### `*_pred_mean` 和 `*_target_mean`

用于观察模型的整体偏置。

例如：

- `counterfactual_pred_mean` 长期接近 0
- `counterfactual_target_mean` 明显偏正

这通常表示模型在把优势差异压平。

---

## 4. Done 指标

### `train/done_acc` / `val/done_acc`

done 分类准确率。

### `train/done_brier` / `val/done_brier`

done 概率的 Brier Score。

### `train/done_prob_mean` / `val/done_prob_mean`

模型平均预测的 done 概率。

### `train/done_target_mean` / `val/done_target_mean`

数据中真实终止样本的比例。

注意：

- 当前 action feature 仍显式包含 `Terminator` 标记
- 因此 done 头可能存在 shortcut

所以 `done_acc` 很高并不一定代表模型真正理解了何时终止。

---

## 5. Valid Action 指标

### `train/valid_f1` / `val/valid_f1`

多标签 F1。

### `train/valid_precision` / `val/valid_precision`

多标签精确率。

### `train/valid_recall` / `val/valid_recall`

多标签召回率。

### `train/valid_label_acc` / `val/valid_label_acc`

逐标签准确率。

### `train/valid_exact_match` / `val/valid_exact_match`

样本级完全匹配率。

### `train/valid_brier` / `val/valid_brier`

多标签概率 Brier Score。

当前解释风险：

- `valid_action_mask` 仍然接近“全 1”
- 因此 `valid_label_acc` 和 `valid_f1` 可能偏乐观

---

## 6. 辅助目标指标

当前默认辅助指标包括：

- `aux_progress_score_mae`
- `aux_coverage_score_mae`
- `aux_conflict_score_mae`
- `aux_redundancy_score_mae`
- `aux_termination_readiness_mae`

这些指标反映模型对 workflow 中间结构状态的拟合能力。

但要注意：

- `uncertainty` target 与 `aux_conflict_score` 仍然同源
- 所以这两类曲线之间会有较强相关性

---

## 7. Counterfactual 指标的当前解释

这是当前版本最需要注意的一组指标。

### `train/counterfactual_mae` / `val/counterfactual_mae`

模型预测的 `q_value()` 与 `credit_targets.leave_one_out_gap` 的平均绝对误差。

### `train/counterfactual_rmse` / `val/counterfactual_rmse`

同一目标的均方根误差。

### `train/counterfactual_pred_mean` / `val/counterfactual_pred_mean`

模型预测的平均 counterfactual 分数。

### `train/counterfactual_target_mean` / `val/counterfactual_target_mean`

数据中的真实平均 counterfactual 分数。

当前 `q_value()` 定义仍是：

```text
reward - cost + gamma * value
```

对应：

- `workflow_world_model.py:776`

因此 current counterfactual loss 的真实含义是：

- 用 `reward - cost + gamma * value` 去逼近“当前边相对兄弟基线的优势”

---

## 8. `val/best_total`

`val/best_total` 表示截至当前 epoch 的最佳验证集总损失，不是当前 epoch 的瞬时值。

它的用途：

- 对齐最佳 checkpoint
- 判断训练是否刷新最优结果

对应：

- tracker 写入：`train_workflow_world_model.py:386`
- checkpoint 保存：`train_workflow_world_model.py:673`

---

## 9. 如何正确解读当前曲线

1. `reward` 下降
   - 说明模型更会拟合树回传的过程价值
   - 不是说明模型更会拟合旧版原始负 reward
2. `value` 下降
   - 当前通常也在说明模型更会拟合同一量纲的树值
3. `counterfactual` 下降
   - 说明模型更会区分“当前分支比兄弟分支更好还是更差”
4. `valid` 很好
   - 依然要先确认你是否接受当前常量式 `valid_action_mask`
5. `done` 很好
   - 依然要注意 Terminator shortcut

---

## 10. 平台说明

虽然文件名叫 `README_wandb.md`，但内容同样适用于 SwanLab。

原因是：

- 两个后端共享同一份指标 payload
- 差别只在日志记录后端，不在指标定义本身
