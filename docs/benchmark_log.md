# Benchmark Log

Use this file to compare generation quality across runs using a fixed prompt set.

## Fixed Prompt Set

1. `Once upon a time`
2. `Lily and Tim went to`
3. `The little cat was`

## Eval Command

```bash
.venv/bin/python -m src.eval
```

## Run Template

Copy this block per run:

```text
Date:
Checkpoint:
Train loss:
Val loss:
Sampling config: temperature=, top_k=, repetition_penalty=, max_new_tokens=

Prompt 1 output:

Prompt 2 output:

Prompt 3 output:

Notes:
```
