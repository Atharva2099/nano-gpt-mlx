# Benchmark Report Template

Use this template to document a complete benchmark snapshot for this TinyStories char-level MLX model.

## 2026-05-01 Char-Level Final Baseline

### Run Metadata

- Date: 2026-05-01
- Run name / tag: `char_baseline`, `seed_sweep`
- Code path: `src/`
- Checkpoint evaluated: `checkpoints/latest.npz`
- Benchmark logs:
  - `docs/benchmarks/20260501_183443_char_baseline.json`
  - `docs/benchmarks/20260501_183515_seed_sweep.json`

### Quantitative Metrics (Latest Checkpoint)

- Params: `533,632` (`0.534M`)
- Train loss: `0.764058`
- Val loss: `0.775700`
- Train perplexity: `2.146972`
- Val perplexity: `2.172113`
- Train bpc: `1.102303`
- Val bpc: `1.119099`
- Train top-1 accuracy: `0.757037`
- Val top-1 accuracy: `0.754084`
- Train eval tokens/sec: `248,593`
- Val eval tokens/sec: `270,663`

### Seed Stability (1337, 42, 7)

- Val loss mean/std: `0.773092 +/- 0.002133`
- Val perplexity mean/std: `2.166460 +/- 0.004622`

### Fixed Prompt Notes

- Outputs are coherent short stories with mild repetition loops.
- Char-level model quality is strong enough to serve as pre-BPE baseline.

## 2026-05-01 BPE-4096 Baseline (8k Steps)

### Run Metadata

- Date: 2026-05-01
- Run name / tag: `bpe4096_8k`, `bpe4096_8k_seed`
- Code path: `src/`
- Checkpoint evaluated: `checkpoints_bpe/latest.npz`
- Benchmark logs:
  - `docs/benchmarks/20260501_190401_bpe4096_8k.json`
  - `docs/benchmarks/20260501_190452_bpe4096_8k_seed.json`

### Quantitative Metrics (Latest Checkpoint)

- Params: `918,592` (`0.919M`)
- Train loss: `2.117203`
- Val loss: `2.239331`
- Train perplexity: `8.307872`
- Val perplexity: `9.387054`
- Train bpc: `3.054479`
- Val bpc: `3.230672`
- Train top-1 accuracy: `0.511924`
- Val top-1 accuracy: `0.497206`
- Train eval tokens/sec: `222,066`
- Val eval tokens/sec: `222,338`

### Seed Stability (1337, 42, 7)

- Val loss mean/std: `2.238943 +/- 0.003707`
- Val perplexity mean/std: `9.383470 +/- 0.034775`

### Fixed Prompt Notes

- Coherence improves quickly and is sentence-level by early training.
- Repetition/template fallback is still visible (`The cat ... The cat ...` pattern).

## 2026-05-01 BPE-1024 Baseline (8k Steps)

### Run Metadata

- Date: 2026-05-01
- Run name / tag: `bpe1024_8k`, `bpe1024_8k_seed`
- Code path: `src/`
- Checkpoint evaluated: `checkpoints_bpe1024/latest.npz`
- Benchmark logs:
  - `docs/benchmarks/20260501_193142_bpe1024_8k.json`
  - `docs/benchmarks/20260501_193227_bpe1024_8k_seed.json`

### Quantitative Metrics (Latest Checkpoint)

- Params: `623,680` (`0.624M`)
- Train loss: `2.020534`
- Val loss: `2.073565`
- Train perplexity: `7.542348`
- Val perplexity: `7.953126`
- Train bpc: `2.915014`
- Val bpc: `2.991522`
- Train top-1 accuracy: `0.527029`
- Val top-1 accuracy: `0.519855`
- Train eval tokens/sec: `219,469`
- Val eval tokens/sec: `239,175`

### Seed Stability (1337, 42, 7)

- Val loss mean/std: `2.078241 +/- 0.004988`
- Val perplexity mean/std: `7.990502 +/- 0.039911`

### Fixed Prompt Notes

- Better quality-per-parameter than BPE-4096 at same step budget.
- Repetition is still present but less degenerate than the BPE-4096 run.

## 2026-05-01 BPE-1024 Extended (16k Steps)

### Run Metadata

- Date: 2026-05-01
- Run name / tag: `bpe1024_16k`, `bpe1024_16k_seed`
- Code path: `src/`
- Checkpoint evaluated: `checkpoints_bpe1024/latest.npz`
- Benchmark logs:
  - `docs/benchmarks/20260501_212148_bpe1024_16k.json`
  - `docs/benchmarks/20260501_212240_bpe1024_16k_seed.json`

### Quantitative Metrics (Latest Checkpoint)

- Params: `623,680` (`0.624M`)
- Train loss: `1.922523`
- Val loss: `1.999487`
- Train perplexity: `6.838189`
- Val perplexity: `7.385269`
- Train bpc: `2.773614`
- Val bpc: `2.884651`
- Train top-1 accuracy: `0.544009`
- Val top-1 accuracy: `0.532850`
- Train eval tokens/sec: `236,753`
- Val eval tokens/sec: `230,267`

### Seed Stability (1337, 42, 7)

- Val loss mean/std: `2.004295 +/- 0.005166`
- Val perplexity mean/std: `7.420956 +/- 0.038393`

### 8k -> 16k Delta (BPE-1024)

- Val loss: `2.073565 -> 1.999487` (improved)
- Val perplexity: `7.953126 -> 7.385269` (improved)
- Val bpc: `2.991522 -> 2.884651` (improved)
- Val top-1: `0.519855 -> 0.532850` (improved)

## Char vs BPE (This Repo Snapshot)

Important comparability rule:
- Perplexity and loss are not directly comparable across different tokenizers.
- Use `bpc` for cross-tokenizer comparison on the same dataset text.

Snapshot comparison:
- Char baseline:
  - params: `0.534M`
  - val bpc: `1.119099`
  - val top-1: `0.754084`
  - val eval tokens/sec: `270,663`
- BPE-4096 baseline:
  - params: `0.919M`
  - val bpc: `3.230672`
  - val top-1: `0.497206`
  - val eval tokens/sec: `222,338`
- BPE-1024 baseline:
  - params: `0.624M`
  - val bpc (8k): `2.991522`
  - val bpc (16k): `2.884651`
  - val top-1 (8k): `0.519855`
  - val top-1 (16k): `0.532850`
  - val eval tokens/sec (8k): `239,175`
  - val eval tokens/sec (16k): `230,267`

Interpretation:
- BPE gave faster qualitative coherence at training time.
- BPE-4096 increased parameter count substantially.
- BPE-1024 is the current winner in this repo snapshot: lower params and better BPE-family metrics.
- Extending BPE-1024 from 8k to 16k clearly improved all tracked quantitative metrics.

## BPE Transition Checklist

1. Freeze char baseline artifacts (done):
   - keep current `checkpoints/latest.npz`
   - keep benchmark JSON logs above
2. Train BPE tokenizer on TinyStories train split.
3. Add tokenizer switch in `src/config.py` and `src/data.py`.
4. Start fresh BPE checkpoints in a new checkpoint directory.
5. Re-run `src.benchmark_suite` with same prompt set and compare:
   - val loss/ppl/bpc/top1
   - throughput
   - repetition/coherence from sample outputs

## BPE Migration Commands

1. Train BPE tokenizer artifact:

```bash
.venv/bin/python -m src.train_tokenizer --train-file data/train.txt --out-dir artifacts/tokenizer_bpe --vocab-size 4096 --min-frequency 2
```

2. Switch config to BPE in `src/config.py`:
   - `tokenizer_type = "bpe"`
   - `bpe_tokenizer_path = "artifacts/tokenizer_bpe/tokenizer.json"`
   - set a new checkpoint directory (recommended): `checkpoint_dir = "checkpoints_bpe"`

3. Start fresh BPE training:

```bash
.venv/bin/python -m src.main
```

4. Run BPE benchmark logs (same harness as char baseline):

```bash
.venv/bin/python -m src.benchmark_suite --ckpt latest.npz --tag bpe_baseline
.venv/bin/python -m src.benchmark_suite --ckpt latest.npz --seeds 1337,42,7 --tag bpe_seed_sweep
```

5. Compare against char baseline:
   - `docs/benchmarks/20260501_183443_char_baseline.json`
   - `docs/benchmarks/20260501_183515_seed_sweep.json`

## Run Metadata

- Date:
- Run name / tag:
- Code path: `src/`
- Checkpoint evaluated: `checkpoints/best.npz` or `checkpoints/latest.npz`

## Model + Training Config

- Tokenizer: character-level
- Vocab size:
- Layers (`n_layers`):
- Model dim (`d_model`):
- Heads (`n_heads`):
- MLP dim (`mlp_dim`):
- Context length:
- Batch size:
- Learning rate:
- Max iters:
- Eval interval:
- Eval iters:

## Quantitative Metrics

- Train loss (final):
- Val loss (final):
- Val perplexity (`exp(val_loss)`):
- Best val loss seen:
- Best val perplexity (`exp(best_val_loss)`):
- Step of best checkpoint:

## Throughput

- Train step time (ms, typical):
- Estimated tokens / sec:
- Hardware:

## Fixed Prompt Outputs

Eval command:

```bash
.venv/bin/python -m src.eval
```

Quantitative + qualitative benchmark logger:

```bash
.venv/bin/python -m src.benchmark_suite --ckpt latest.npz --tag char_baseline
```

Seed sweep example:

```bash
.venv/bin/python -m src.benchmark_suite --ckpt latest.npz --seeds 1337,42,7 --tag seed_sweep
```

Context/model sweep examples (requires matching checkpoints):

```bash
.venv/bin/python -m src.benchmark_suite --ckpt latest.npz --contexts 128,192 --tag context_sweep
.venv/bin/python -m src.benchmark_suite --ckpt latest.npz --d-models 96,128 --tag dmodel_sweep
```

Prompt 1 (`Once upon a time`):

Prompt 2 (`Lily and Tim went to`):

Prompt 3 (`The little cat was`):

## Qualitative Notes

- Coherence:
- Repetition:
- Failure modes:
- Improvement over prior run:

## Reproducibility

- Seed:
- Config file: `src/config.py`
- Data files:
  - `data/train.txt`
  - `data/val.txt`
- Checkpoint files:
  - `checkpoints/best.npz`
  - `checkpoints/latest.npz`
  - `checkpoints/latest_step.txt`
  - `checkpoints/best_val.txt`
  - `checkpoints/latest_optim.npz`
