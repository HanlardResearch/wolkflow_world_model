# World Model Dataset Conflict Report

## Overview

- Files analyzed: 96
- Records analyzed: 771
- Unique (task_type, workflow_state, action) signatures: 250
- Repeated signatures: 80 (32.00%)

## Warnings

- reward has high repeated-signature conflict ratio (383/771 = 49.68%).
- value has high repeated-signature conflict ratio (383/771 = 49.68%).
- cost has high repeated-signature conflict ratio (454/771 = 58.88%).
- counterfactual has high repeated-signature conflict ratio (209/771 = 27.11%).

## Label Conflicts

### reward

- Conflicting signatures: 31 / 80 (38.75%)
- Conflicting records: 383 / 771 (49.68%)
- Conflict range mean/p50/p75/max: 1.9194 / 2.0000 / 2.0000 / 2.0000

- Example conflicting signatures:
- ReasoningAgent_gpt4o @ ((None, None, -1),) count=81 range=2.0000 values=[-1.0, -0.5, 1.0]
- ConcluderAgent_gpt4o @ (('ReasoningAgent_gpt4o', 'reasoning', 1),) count=24 range=2.0000 values=[-1.0, 1.0]
- CriticAgent_gpt4o @ (('ReasoningAgent_gpt4o', 'reasoning', 1),) count=23 range=2.0000 values=[-1.0, 1.0]
- TavilyAgent @ ((None, None, -1),) count=23 range=2.0000 values=[-1.0, 1.0]
- TerminatorAgent @ (('ReasoningAgent_gpt4o', 'reasoning', 1), ('CriticAgent_gpt4o', 'critique', 1)) count=21 range=2.0000 values=[-1.0, 1.0]
- TerminatorAgent @ (('ReasoningAgent_gpt4o', 'reasoning', 1), ('ConcluderAgent_gpt4o', 'conclude', 1)) count=21 range=2.0000 values=[-1.0, 1.0]
- QuestionAgent_gpt4o @ ((None, None, -1),) count=18 range=2.0000 values=[-1.0, 1.0]
- FileAgent @ ((None, None, -1),) count=14 range=2.0000 values=[-1.0, 1.0]

### value

- Conflicting signatures: 31 / 80 (38.75%)
- Conflicting records: 383 / 771 (49.68%)
- Conflict range mean/p50/p75/max: 1.9194 / 2.0000 / 2.0000 / 2.0000

- Example conflicting signatures:
- ReasoningAgent_gpt4o @ ((None, None, -1),) count=81 range=2.0000 values=[-1.0, -0.5, 1.0]
- ConcluderAgent_gpt4o @ (('ReasoningAgent_gpt4o', 'reasoning', 1),) count=24 range=2.0000 values=[-1.0, 1.0]
- CriticAgent_gpt4o @ (('ReasoningAgent_gpt4o', 'reasoning', 1),) count=23 range=2.0000 values=[-1.0, 1.0]
- TavilyAgent @ ((None, None, -1),) count=23 range=2.0000 values=[-1.0, 1.0]
- TerminatorAgent @ (('ReasoningAgent_gpt4o', 'reasoning', 1), ('CriticAgent_gpt4o', 'critique', 1)) count=21 range=2.0000 values=[-1.0, 1.0]
- TerminatorAgent @ (('ReasoningAgent_gpt4o', 'reasoning', 1), ('ConcluderAgent_gpt4o', 'conclude', 1)) count=21 range=2.0000 values=[-1.0, 1.0]
- QuestionAgent_gpt4o @ ((None, None, -1),) count=18 range=2.0000 values=[-1.0, 1.0]
- FileAgent @ ((None, None, -1),) count=14 range=2.0000 values=[-1.0, 1.0]

### cost

- Conflicting signatures: 52 / 80 (65.00%)
- Conflicting records: 454 / 771 (58.88%)
- Conflict range mean/p50/p75/max: 0.0458 / 0.0412 / 0.0595 / 0.1323

- Example conflicting signatures:
- ReasoningAgent_gpt4o @ (('PythonAgent_gpt4o', 'run_python', 0),) count=42 range=0.1323 values=[0.792017, 0.813965, 0.823788, 0.824172, 0.825377, 0.82819, 0.831206, 0.832679, 0.83611, 0.838412, 0.839521, 0.840829]
- ReasoningAgent_gpt4o @ ((None, None, -1),) count=81 range=0.1300 values=[0.802138, 0.803395, 0.804917, 0.806627, 0.810495, 0.811111, 0.813117, 0.813415, 0.817629, 0.818754, 0.825409, 0.826055]
- PythonAgent_gpt4o @ ((None, None, -1),) count=51 range=0.1207 values=[0.741585, 0.74428, 0.748414, 0.754778, 0.755352, 0.756673, 0.758542, 0.760114, 0.760186, 0.760955, 0.761114, 0.779163]
- TavilyAgent @ ((None, None, -1),) count=23 range=0.1200 values=[0.77437, 0.792771, 0.799247, 0.80197, 0.808911, 0.810505, 0.826624, 0.827306, 0.828402, 0.830482, 0.834115, 0.841063]
- FileAgent @ ((None, None, -1),) count=14 range=0.0991 values=[0.78307, 0.807295, 0.813216, 0.833771, 0.834673, 0.844124, 0.866519, 0.870534, 0.882121]
- PythonAgent_gpt4o @ (('PlannerAgent_gpt4o', 'planning', 1),) count=5 range=0.0913 values=[0.757587, 0.784627, 0.848877]
- ReasoningAgent_gpt4o @ (('PythonAgent_gpt4o', 'run_python', 0), ('PythonAgent_gpt4o', 'run_python', 0)) count=3 range=0.0895 values=[0.823794, 0.832108, 0.913334]
- ConcluderAgent_gpt4o @ (('TavilyAgent', 'search_tavily', 1),) count=6 range=0.0799 values=[0.814768, 0.828035, 0.847218, 0.863303, 0.894699]

### uncertainty

- Conflicting signatures: 1 / 80 (1.25%)
- Conflicting records: 8 / 771 (1.04%)
- Conflict range mean/p50/p75/max: 1.0000 / 1.0000 / 1.0000 / 1.0000

- Example conflicting signatures:
- ArxivAgent @ ((None, None, -1),) count=8 range=1.0000 values=[0.0, 1.0]

### counterfactual

- Conflicting signatures: 9 / 80 (11.25%)
- Conflicting records: 209 / 771 (27.11%)
- Conflict range mean/p50/p75/max: 1.8333 / 2.0000 / 2.0000 / 4.0000

- Example conflicting signatures:
- ReasoningAgent_gpt4o @ ((None, None, -1),) count=81 range=4.0000 values=[-2.0, 0.0, 2.0]
- QuestionAgent_gpt4o @ ((None, None, -1),) count=18 range=3.0000 values=[-2.0, 0.0, 1.0]
- TavilyAgent @ ((None, None, -1),) count=23 range=2.0000 values=[0.0, 1.0, 2.0]
- FileAgent @ ((None, None, -1),) count=14 range=2.0000 values=[-2.0, 0.0]
- TavilyAgent @ (('ReasoningAgent_gpt4o', 'reasoning', 1),) count=7 range=2.0000 values=[-2.0, 0.0]
- PythonAgent_gpt4o @ ((None, None, -1),) count=51 range=1.0000 values=[0.0, 1.0]
- ArxivAgent @ ((None, None, -1),) count=8 range=1.0000 values=[0.0, 1.0]
- QuestionAgent_gpt4o @ (('PythonAgent_gpt4o', 'run_python', 0),) count=4 range=1.0000 values=[-1.0, 0.0]

### done

- Conflicting signatures: 0 / 80 (0.00%)
- Conflicting records: 0 / 771 (0.00%)
- Conflict range mean/p50/p75/max: 0.0000 / 0.0000 / 0.0000 / 0.0000

- Example conflicting signatures:
- None
