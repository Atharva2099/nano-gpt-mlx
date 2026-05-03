# nano-gpt-mlx

A minimal GPT-2 style decoder-only language model built entirely in **MLX** for Apple Silicon. Designed to train from scratch on an M1 Mac with 8GB RAM.

## Overview

This project implements a complete LLM training pipeline in pure MLX — no PyTorch, TensorFlow, or JAX. It includes:

- **Pretraining** on the TinyStories dataset (unsupervised next-token prediction)
- **Supervised Fine-Tuning (SFT)** on TinyStoriesInstruct with prompt masking
- **BPE tokenization** with configurable vocab sizes
- **Generation** with temperature, top-k, and repetition penalty sampling

## Model Architecture

| Spec | Value |
|------|-------|
| Parameters | ~3.5M (current) |
| Layers | 6 |
| d_model | 192 |
| Heads | 6 |
| MLP dim | 768 |
| Context length | 256 |
| Vocab | 4096 (BPE) |

Architecture: GPT-2 style decoder-only transformer with tied input/output embeddings, causal self-attention, and pre-LayerNorm.

## Project Structure

```
.
├── src/
│   ├── config.py           # Hyperparameters
│   ├── model.py            # NanoGPT transformer (MLX)
│   ├── data.py             # TinyStories download & tokenization
│   ├── train.py            # Pretraining loop
│   ├── main.py             # Pretraining entry point
│   ├── train_tokenizer.py  # BPE tokenizer training
│   ├── prepare_sft_data.py # SFT dataset prep
│   ├── sft_data.py         # SFT batching with prompt masking
│   ├── sft_train.py        # SFT training loop
│   ├── sft_main.py         # SFT entry point
│   ├── sft_eval.py         # SFT evaluation
│   ├── eval.py             # Pretrain evaluation
│   ├── benchmark.py        # Perplexity benchmarks
│   └── benchmark_suite.py  # Full benchmark suite
├── artifacts/              # Tokenizers
├── data/                   # Datasets (not in repo)
└── checkpoints*/           # Model weights (not in repo)
```

## Requirements

- macOS with Apple Silicon
- Python 3.10+
- `uv` (recommended) or `pip`

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Quick Start

### 1. Train Tokenizer

```bash
uv run python src/train_tokenizer.py --out-dir artifacts/tokenizer_bpe4096 --vocab-size 4096
```

### 2. Pretrain

```bash
uv run python -m src.main
```

Loads automatically from `checkpoints_bpe4096/latest.npz` if resuming. Best checkpoint saved to `best.npz`.

### 3. Prepare SFT Data

```bash
uv run python src/prepare_sft_data.py --max-samples 300000
```

### 4. SFT

```bash
uv run python -m src.sft_main
```

Loads pretrained weights first, then fine-tunes with prompt masking.

### 5. Evaluate

```bash
# Pretrain generation
uv run python -m src.eval

# SFT generation
uv run python -m src.sft_eval
```

## Training Details

| Stage | Dataset | Examples | Iters | LR | Batch |
|-------|---------|----------|-------|-----|-------|
| Pretrain | TinyStories | ~20k stories | 16,000 | 1e-3 | 16 |
| SFT | TinyStoriesInstruct | ~270k pairs | 10,000 | 1e-4 | 8 |

SFT uses **prompt masking** — loss is only computed on response tokens, not the instruction prompt.

## Model Evolution

This project has gone through several iterations as we scaled up vocabulary size, parameters, and context length. All versions are preserved locally in `checkpoints*_v1/` directories.

| Version | Tokenizer | Params | d_model | Layers | Context | Pretrain Loss | SFT Loss | Notes |
|---------|-----------|--------|---------|--------|---------|---------------|----------|-------|
| **v4 (current)** | BPE 4096 | **3.5M** | 192 | 6 | 256 | 2.00 | **1.82** | Best quality. Diverse prompts, real instruction-following. |
| v3 | BPE 4096 | 0.95M | 96 | 4 | 128 | 2.16 | 2.20 | Larger vocab but same tiny model. Still repetitive. |
| v2 | BPE 1024 | 0.62M | 96 | 4 | 128 | 2.00 | 2.01 | First working SFT. Subword loops ("machones", "sungony"). |
| v1.5 | BPE 256 | 0.47M | 96 | 4 | 128 | 2.24 | — | Too small vocab, poor quality. |
| v1 | Char-level | 0.47M | 96 | 4 | 128 | 0.77 | — | Baseline. Char vocab=~65. Not comparable loss. |

### Key Lessons

- **Vocab size matters:** 256→1024→4096 eliminated gibberish subword generation.
- **Parameters matter more:** 0.6M→3.5M was a bigger quality jump than any vocab change. Repetition loops disappeared.
- **Context matters:** 128→256 allowed multi-sentence coherence.
- **Prompt diversity matters:** Early SFT used only `"Write a short story."` for all 270k examples. Fixing `prepare_sft_data.py` to extract real constraints (words, features, summaries) made the model actually follow instructions.

## Design Philosophy

- **MLX-native**: Every tensor op, autograd call, module, optimizer, and training loop uses MLX
- **Minimal**: No bloated abstractions or excessive guardrails
- **Readable**: Straightforward code that maps 1:1 to the architecture diagram
- **Small**: Fits and trains on commodity hardware (8GB M1 Mac)

## Benchmarks

Run the benchmark suite after pretraining:

```bash
uv run python -m src.benchmark_suite
```

## Limitations

- 256-token context limits coherence for longer stories
- 3.5M parameters is still tiny — character names can drift, tense can mix
- Trained only on TinyStories — no general knowledge or reasoning
- Designed as an educational baseline, not a production model

## License

MIT
