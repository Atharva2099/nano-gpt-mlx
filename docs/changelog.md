# Changelog

## 2026-05-01

### Changed
- Updated generation sampling in `model.py`:
  - from `mx.random.categorical(mx.softmax(logits, axis=-1))`
  - to `mx.random.categorical(logits, axis=-1)`

### Reason
- MLX categorical sampling expects logits.

### Result
- Generated text became more readable at similar training stages.

### Changed
- Added checkpoint support and best-validation tracking:
  - `src/train.py` now saves `checkpoints/latest.npz`, `checkpoints/best.npz`,
    `checkpoints/latest_step.txt`, and `checkpoints/best_val.txt`.
  - `src/main.py` now auto-loads `checkpoints/latest.npz` when present.

### Changed
- Added generation controls:
  - `src/model.py`: `generate(..., top_k=None)` with top-k logits filtering.
  - `src/train.py`: uses configurable sampling controls and prints multiple eval samples.
  - `src/config.py`: added `sample_temperature`, `sample_top_k`, `sample_count`,
    `sample_max_new_tokens`, and `sample_prompt`.

### Changed
- Added prompt-diversified eval sampling:
  - `src/config.py`: added `sample_prompts` list.
  - `src/train.py`: cycles through prompt list and prints prompt labels with samples.

### Changed
- Added repetition penalty in decoding:
  - `src/model.py`: `generate(..., repetition_penalty=1.0)` penalizes logits for already-seen tokens.

### Changed
- Moved all project code under `src/` and updated entrypoint usage:
  - training: `.venv/bin/python -m src.main`
  - eval: `.venv/bin/python -m src.eval`

### Changed
- Added benchmark suite logger:
  - `src/benchmark_suite.py` logs loss, perplexity, bpc, top-1 accuracy,
    throughput, and prompt samples to `docs/benchmarks/*.json`.

### Changed
- Recorded char-level final baseline and BPE transition checklist:
  - `docs/benchmark_report.md` now includes concrete 2026-05-01 baseline metrics
    and next-step criteria for BPE migration.

### Changed
- Added BPE tokenizer training entrypoint:
  - `src/train_tokenizer.py` trains and saves `artifacts/tokenizer_bpe/tokenizer.json`.

### Changed
- Added tokenizer mode switch in data pipeline:
  - `src/config.py`: `tokenizer_type` and `bpe_tokenizer_path`
  - `src/data.py`: char/BPE tokenization switch using config

### Changed
- Recorded BPE-1024 benchmark runs and comparison outcome:
  - added `bpe1024_8k` and `bpe1024_8k_seed` metrics to `docs/benchmark_report.md`
  - documented BPE-1024 as current winner vs BPE-4096 for this training budget

### Changed
- Recorded BPE-1024 extended training benchmarks:
  - added `bpe1024_16k` and `bpe1024_16k_seed` metrics to `docs/benchmark_report.md`
  - documented 8k -> 16k improvements and updated current-best baseline
