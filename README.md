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
| Parameters | ~0.6M (with 4096 BPE vocab) |
| Layers | 4 |
| d_model | 96 |
| Heads | 4 |
| MLP dim | 384 |
| Context length | 128 |
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
| Pretrain | TinyStories | ~20k stories | 16,000 | 1e-3 | 32 |
| SFT | TinyStoriesInstruct | ~270k pairs | 10,000 | 1e-4 | 16 |

SFT uses **prompt masking** — loss is only computed on response tokens, not the instruction prompt.

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

- 128-token context limits coherence for longer stories
- 0.6M parameters produces simple, repetitive text
- Trained only on TinyStories — no general knowledge

## License

MIT
