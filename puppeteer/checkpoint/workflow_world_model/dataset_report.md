# Dataset Split Report

## Overview

- Train files: 436
- Validation files: 10
- Train records: 3387
- Validation records: 48

## Train

- Episodes: 434
- Episode length mean/std/min/max: 7.8041 / 5.4486 / 2.0000 / 24.0000
- Reward mean/std/min/max: 0.7673 / 0.6318 / -1.0000 / 1.0000
- Value mean/std/min/max: 0.7673 / 0.6318 / -1.0000 / 1.0000
- Uncertainty mean/std: 0.2460 / 0.3153
- Valid action count mean/std: 14.0000 / 0.0000
- Done rate: 0.2707
- Next-state present rate: 1.0000
- Next-graph present rate: 1.0000
- Returns present rate: 1.0000
- Next-targets present rate: 1.0000
- Top actions: TerminatorAgent(917), ReasoningAgent_gpt4o(912), PythonAgent_gpt4o(426), ConcluderAgent_gpt4o(404), CriticAgent_gpt4o(289), TavilyAgent(138), QuestionAgent_gpt4o(121), FileAgent(69)

## Val

- Episodes: 10
- Episode length mean/std/min/max: 4.8000 / 2.8914 / 2.0000 / 12.0000
- Reward mean/std/min/max: 0.7500 / 0.6614 / -1.0000 / 1.0000
- Value mean/std/min/max: 0.7500 / 0.6614 / -1.0000 / 1.0000
- Uncertainty mean/std: 0.1198 / 0.2684
- Valid action count mean/std: 14.0000 / 0.0000
- Done rate: 0.3333
- Next-state present rate: 1.0000
- Next-graph present rate: 1.0000
- Returns present rate: 1.0000
- Next-targets present rate: 1.0000
- Top actions: TerminatorAgent(16), ReasoningAgent_gpt4o(14), TavilyAgent(5), ConcluderAgent_gpt4o(5), CriticAgent_gpt4o(4), PythonAgent_gpt4o(3), QuestionAgent_gpt4o(1)

## Comparison

- Reward mean gap (val-train): -0.0173
- Value mean gap (val-train): -0.0173
- Cost target mean gap (val-train): -0.0477
- Uncertainty mean gap (val-train): -0.1262
- Done rate gap (val-train): 0.0626
- Validation action coverage by training action vocab: 1.0000

## Warnings

- Uncertainty/conflict targets differ noticeably between train and val (gap=-0.1262).

## Target Diagnostics

### reward

- Kind: regression
- Train mean/std/p50/unique_ratio/dominant_ratio: 0.7673 / 0.6318 / 1.0000 / 0.0021 / 0.8769
- Val mean/std/p50/unique_ratio/dominant_ratio: 0.7500 / 0.6614 / 1.0000 / 0.0417 / 0.8750
- Mean gap (val-train): -0.0173

### value

- Kind: regression
- Train mean/std/p50/unique_ratio/dominant_ratio: 0.7673 / 0.6318 / 1.0000 / 0.0021 / 0.8769
- Val mean/std/p50/unique_ratio/dominant_ratio: 0.7500 / 0.6614 / 1.0000 / 0.0417 / 0.8750
- Mean gap (val-train): -0.0173

### cost_target

- Kind: regression
- Train mean/std/p50/unique_ratio/dominant_ratio: 0.6110 / 0.3735 / 0.8209 / 0.5991 / 0.2707
- Val mean/std/p50/unique_ratio/dominant_ratio: 0.5633 / 0.3990 / 0.8216 / 0.6458 / 0.3333
- Mean gap (val-train): -0.0477

### uncertainty

- Kind: regression
- Train mean/std/p50/unique_ratio/dominant_ratio: 0.2460 / 0.3153 / 0.0000 / 0.0030 / 0.5125
- Val mean/std/p50/unique_ratio/dominant_ratio: 0.1198 / 0.2684 / 0.0000 / 0.1042 / 0.7917
- Mean gap (val-train): -0.1262

### counterfactual

- Kind: regression
- Train mean/std/p50/unique_ratio/dominant_ratio: 0.0000 / 0.2263 / 0.0000 / 0.0032 / 0.9761
- Val mean/std/p50/unique_ratio/dominant_ratio: 0.0000 / 0.4082 / 0.0000 / 0.0625 / 0.9583
- Mean gap (val-train): 0.0000

### done

- Kind: binary
- Train mean/std/p50/unique_ratio/dominant_ratio: 0.2707 / 0.4443 / 0.0000 / 0.0006 / 0.7293
- Val mean/std/p50/unique_ratio/dominant_ratio: 0.3333 / 0.4714 / 0.0000 / 0.0417 / 0.6667
- Mean gap (val-train): 0.0626

### valid_action_count

- Kind: count
- Train mean/std/p50/unique_ratio/dominant_ratio: 14.0000 / 0.0000 / 14.0000 / 0.0003 / 1.0000
- Val mean/std/p50/unique_ratio/dominant_ratio: 14.0000 / 0.0000 / 14.0000 / 0.0208 / 1.0000
- Mean gap (val-train): 0.0000

### Target Warnings

- reward/train has very low label diversity (0.0021).
- reward/val has very low label diversity (0.0417).
- value/train has very low label diversity (0.0021).
- value/val has very low label diversity (0.0417).
- uncertainty/train has very low label diversity (0.0030).
- counterfactual/train is dominated by one value (0.9761).
- counterfactual/train has very low label diversity (0.0032).
- counterfactual/val is dominated by one value (0.9583).

## Label Conflicts

### reward

- train: conflicting groups=88/214 (0.4112)
- train: conflicting records=2506 (0.7399)
- val: conflicting groups=3/11 (0.2727)
- val: conflicting records=15 (0.3125)
- Example signatures:
  - ReasoningAgent_gpt4o @ ((None, None, -1),) count=9 values=[-1.0, 1.0]
  - ConcluderAgent_gpt4o @ (('ReasoningAgent_gpt4o', 'reasoning', 1),) count=3 values=[-1.0, 1.0]
  - TerminatorAgent @ (('ReasoningAgent_gpt4o', 'reasoning', 1), ('ConcluderAgent_gpt4o', 'conclude', 1)) count=3 values=[-1.0, 1.0]

### value

- train: conflicting groups=88/214 (0.4112)
- train: conflicting records=2506 (0.7399)
- val: conflicting groups=3/11 (0.2727)
- val: conflicting records=15 (0.3125)
- Example signatures:
  - ReasoningAgent_gpt4o @ ((None, None, -1),) count=9 values=[-1.0, 1.0]
  - ConcluderAgent_gpt4o @ (('ReasoningAgent_gpt4o', 'reasoning', 1),) count=3 values=[-1.0, 1.0]
  - TerminatorAgent @ (('ReasoningAgent_gpt4o', 'reasoning', 1), ('ConcluderAgent_gpt4o', 'conclude', 1)) count=3 values=[-1.0, 1.0]

### Conflict Warnings

- reward/train has conflicting labels for repeated state-action signatures (record_ratio=0.7399, group_ratio=0.4112).
- reward/val has conflicting labels for repeated state-action signatures (record_ratio=0.3125, group_ratio=0.2727).
- value/train has conflicting labels for repeated state-action signatures (record_ratio=0.7399, group_ratio=0.4112).
- value/val has conflicting labels for repeated state-action signatures (record_ratio=0.3125, group_ratio=0.2727).
