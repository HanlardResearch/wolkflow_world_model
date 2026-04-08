# World Model Analysis

## Overview

- Checkpoint: checkpoint/workflow_world_model/best_world_model.pt
- Validation records analyzed: 48
- Checkpoint epoch: 43
- Stored best val total: 0.3131

## Key Metrics

- aux_skill_mean: 0.7414
- cost_skill: 0.9954
- done_acc: 1.0000
- overall: 0.8491
- reward_skill: 0.4942
- uncertainty_skill: 0.9850
- valid_exact_match: 1.0000
- valid_f1: 1.0000
- value_skill: 0.5766

## Head Summary

- reward: reward: mae=0.3938 rmse=0.4704 pred=0.5685 target=0.7500
- cost: cost: mae=0.0171 rmse=0.0269 pred=0.5578 target=0.5633
- value: value: mae=0.3138 rmse=0.4304 pred=0.6494 target=0.7500
- uncertainty: uncertainty: mae=0.0180 rmse=0.0329 pred=0.1293 target=0.1198
- done: acc=1.0000 brier=0.0000
- valid: f1=1.0000 exact=1.0000

## Strengths

- overall key metric is strong (0.849).
- cost head generalizes well (skill=0.995).
- uncertainty head generalizes well (skill=0.985).
- done prediction is reliable (acc=1.000, brier=0.000).
- valid-action prediction is nearly exact (f1=1.000, exact=1.000).

## Weaknesses

- no obvious failure head was detected from the current validation metrics.

## Worst Cases

### reward

- idx=16 action=TerminatorAgent state=(('ReasoningAgent_gpt4o', 'reasoning', 1), ('ConcluderAgent_gpt4o', 'conclude', 1)) target=-1.0000 pred=0.4270 abs_error=1.4270
- idx=14 action=ReasoningAgent_gpt4o state=((None, None, -1),) target=-1.0000 pred=0.4254 abs_error=1.4254
- idx=15 action=ConcluderAgent_gpt4o state=(('ReasoningAgent_gpt4o', 'reasoning', 1),) target=-1.0000 pred=0.1670 abs_error=1.1670
- idx=1 action=TavilyAgent state=(('ReasoningAgent_gpt4o', 'reasoning', 1),) target=1.0000 pred=0.3782 abs_error=0.6218
- idx=26 action=QuestionAgent_gpt4o state=((None, None, -1),) target=-1.0000 pred=-0.4288 abs_error=0.5712

### cost

- idx=42 action=PythonAgent_gpt4o state=((None, None, -1),) target=0.8514 pred=0.7622 abs_error=0.0892
- idx=32 action=PythonAgent_gpt4o state=((None, None, -1),) target=0.8623 pred=0.7741 abs_error=0.0882
- idx=33 action=ReasoningAgent_gpt4o state=(('PythonAgent_gpt4o', 'run_python', 0),) target=0.9243 pred=0.8474 abs_error=0.0769
- idx=29 action=ReasoningAgent_gpt4o state=((None, None, -1),) target=0.8049 pred=0.8450 abs_error=0.0401
- idx=39 action=ReasoningAgent_gpt4o state=(('PythonAgent_gpt4o', 'run_python', 0),) target=0.8693 pred=0.8310 abs_error=0.0384

### value

- idx=14 action=ReasoningAgent_gpt4o state=((None, None, -1),) target=-1.0000 pred=0.4868 abs_error=1.4868
- idx=16 action=TerminatorAgent state=(('ReasoningAgent_gpt4o', 'reasoning', 1), ('ConcluderAgent_gpt4o', 'conclude', 1)) target=-1.0000 pred=0.4759 abs_error=1.4759
- idx=15 action=ConcluderAgent_gpt4o state=(('ReasoningAgent_gpt4o', 'reasoning', 1),) target=-1.0000 pred=0.1869 abs_error=1.1869
- idx=26 action=QuestionAgent_gpt4o state=((None, None, -1),) target=-1.0000 pred=-0.3373 abs_error=0.6627
- idx=1 action=TavilyAgent state=(('ReasoningAgent_gpt4o', 'reasoning', 1),) target=1.0000 pred=0.4413 abs_error=0.5587

### uncertainty

- idx=27 action=ConcluderAgent_gpt4o state=(('QuestionAgent_gpt4o', 'question', 1),) target=0.0000 pred=0.1709 abs_error=0.1709
- idx=34 action=TerminatorAgent state=(('PythonAgent_gpt4o', 'run_python', 0), ('ReasoningAgent_gpt4o', 'reasoning', 1)) target=0.3333 pred=0.2758 abs_error=0.0576
- idx=28 action=TerminatorAgent state=(('QuestionAgent_gpt4o', 'question', 1), ('ConcluderAgent_gpt4o', 'conclude', 1)) target=0.0000 pred=0.0569 abs_error=0.0569
- idx=25 action=TerminatorAgent state=(('ReasoningAgent_gpt4o', 'reasoning', 1), ('CriticAgent_gpt4o', 'critique', 1)) target=0.0000 pred=0.0487 abs_error=0.0487
- idx=40 action=ConcluderAgent_gpt4o state=(('PythonAgent_gpt4o', 'run_python', 0), ('ReasoningAgent_gpt4o', 'reasoning', 1)) target=0.3333 pred=0.2851 abs_error=0.0482

### done

- idx=26 action=QuestionAgent_gpt4o state=((None, None, -1),) target=0.0000 pred=0.0010 wrong=0 gap=0.0010
- idx=25 action=TerminatorAgent state=(('ReasoningAgent_gpt4o', 'reasoning', 1), ('CriticAgent_gpt4o', 'critique', 1)) target=1.0000 pred=0.9996 wrong=0 gap=0.0004
- idx=22 action=TerminatorAgent state=(('TavilyAgent', 'search_tavily', 1), ('ReasoningAgent_gpt4o', 'reasoning', 1)) target=1.0000 pred=0.9999 wrong=0 gap=0.0001
- idx=28 action=TerminatorAgent state=(('QuestionAgent_gpt4o', 'question', 1), ('ConcluderAgent_gpt4o', 'conclude', 1)) target=1.0000 pred=0.9999 wrong=0 gap=0.0001
- idx=19 action=TerminatorAgent state=(('ReasoningAgent_gpt4o', 'reasoning', 1), ('CriticAgent_gpt4o', 'critique', 1)) target=1.0000 pred=1.0000 wrong=0 gap=0.0000

### valid

- idx=0 action=ReasoningAgent_gpt4o state=((None, None, -1),) target_count=14 pred_count=14 mismatch=0 exact=1
- idx=1 action=TavilyAgent state=(('ReasoningAgent_gpt4o', 'reasoning', 1),) target_count=14 pred_count=14 mismatch=0 exact=1
- idx=2 action=TerminatorAgent state=(('ReasoningAgent_gpt4o', 'reasoning', 1), ('TavilyAgent', 'search_tavily', 1)) target_count=14 pred_count=14 mismatch=0 exact=1
- idx=3 action=TavilyAgent state=((None, None, -1),) target_count=14 pred_count=14 mismatch=0 exact=1
- idx=4 action=TavilyAgent state=(('TavilyAgent', 'search_tavily', 1),) target_count=14 pred_count=14 mismatch=0 exact=1
