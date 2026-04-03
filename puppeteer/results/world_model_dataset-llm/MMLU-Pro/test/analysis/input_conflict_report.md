# World Model Input Conflict Report

## Overview

- Files analyzed: 1
- Records analyzed: 6
- Unique coarse signatures: 6
- Unique model input hashes: 6

## Key Finding

This report compares two notions of `same x`:

- `coarse_signature`: `(task_type, workflow_state, action_name)`
- `model_input_hash`: the actual current model-visible batch tensors hashed row by row

- reward: coarse conflict=0.00%, model-input conflict=0.00%, relative reduction=0.00%
- value: coarse conflict=0.00%, model-input conflict=0.00%, relative reduction=0.00%

## Warnings

- No major conflicts detected.

## coarse_signature

- Repeated groups: 0 / 6 (0.00%)
- Repeated records: 0 / 6 (0.00%)

### reward

- Conflicting groups: 0 / 0 (0.00%)
- Conflicting records: 0 / 6 (0.00%)
- Conflict range mean/p50/p75/max: 0.0000 / 0.0000 / 0.0000 / 0.0000

- Example groups:
- None
### value

- Conflicting groups: 0 / 0 (0.00%)
- Conflicting records: 0 / 6 (0.00%)
- Conflict range mean/p50/p75/max: 0.0000 / 0.0000 / 0.0000 / 0.0000

- Example groups:
- None

## model_input_hash

- Repeated groups: 0 / 6 (0.00%)
- Repeated records: 0 / 6 (0.00%)

### reward

- Conflicting groups: 0 / 0 (0.00%)
- Conflicting records: 0 / 6 (0.00%)
- Conflict range mean/p50/p75/max: 0.0000 / 0.0000 / 0.0000 / 0.0000

- Example groups:
- None
### value

- Conflicting groups: 0 / 0 (0.00%)
- Conflicting records: 0 / 6 (0.00%)
- Conflict range mean/p50/p75/max: 0.0000 / 0.0000 / 0.0000 / 0.0000

- Example groups:
- None
