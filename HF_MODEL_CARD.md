---
library_name: mlx
language:
  - en
tags:
  - mlx
  - gpt2
  - decoder-only
  - tinystories
  - apple-silicon
  - from-scratch
license: mit
---

# nano-gpt-mlx-0.6M

A 0.6M parameter GPT-2 style decoder-only language model trained from scratch in pure MLX on Apple Silicon.

## Model Description

This is a minimal transformer language model designed to run and train on an M1 Mac with 8GB RAM. It was trained in two stages:

1. **Pretraining** on TinyStories (next-token prediction)
2. **Supervised Fine-Tuning** on TinyStoriesInstruct (instruction-following with prompt masking)

All training code is implemented in MLX only — no PyTorch, TensorFlow, or JAX.

### Architecture

| Spec | Value |
|------|-------|
| Parameters | ~0.6M |
| Layers | 4 |
| d_model | 96 |
| Attention heads | 4 |
| MLP dimension | 384 |
| Context length | 128 |
| Vocabulary | 4096 (BPE) |
| Embeddings | Tied (input/output) |

## Training

### Pretraining
- **Dataset**: [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
- **Objective**: Causal language modeling (next-token prediction)
- **Steps**: 16,000
- **Learning rate**: 1e-3
- **Batch size**: 32
- **Optimizer**: AdamW

### Supervised Fine-Tuning
- **Dataset**: [roneneldan/TinyStoriesInstruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct) (300k subset)
- **Objective**: Causal LM with prompt masking (loss only on response tokens)
- **Steps**: 10,000
- **Learning rate**: 1e-4
- **Batch size**: 16
- **Optimizer**: AdamW

## Usage

### Load in MLX

```python
import mlx.core as mx
from tokenizers import Tokenizer

# Load tokenizer
tokenizer = Tokenizer.from_file("tokenizer/tokenizer.json")

# Load model weights
weights = mx.load("model.npz")

# Build model and load weights
from src.model import NanoGPT
from src import config

model = NanoGPT(
    tokenizer.get_vocab_size(),
    config.d_model,
    config.n_heads,
    config.n_layers,
    config.mlp_dim,
    config.context_length,
)
model.load_weights("model.npz")
model.eval()

# Generate
prompt = "Write a short story."
ids = tokenizer.encode(prompt).ids
context = mx.array([ids], dtype=mx.int32)
output = model.generate(context, max_new_tokens=100, temperature=0.8, top_k=20)
decoded = tokenizer.decode(output[0].tolist())
print(decoded)
```

### Load from source

Clone the full training repo:

```bash
git clone https://github.com/Atharva2099/nano-gpt-mlx.git
cd nano-gpt-mlx
uv run python -m src.sft_eval
```

## Evaluation

This model achieves ~2.0 cross-entropy loss on the TinyStoriesInstruct holdout set after SFT. Given the 4096-token BPE vocabulary, this corresponds to reasonable fluency for simple stories.

### Perplexity (estimated)

| Split | PPL |
|-------|-----|
| TinyStories val | ~6-7 |
| TinyStoriesInstruct test | ~7-8 |

## Limitations

- **Tiny scale**: 0.6M parameters is extremely small. Output is simple and can become repetitive.
- **Short context**: 128 tokens limits coherent generation to brief passages.
- **Narrow domain**: Trained only on children's stories. No general knowledge, reasoning, or factual accuracy.
- **Hardware-constrained**: Designed for 8GB M1 Mac; larger models would improve quality significantly.

## Intended Use

- Educational demonstration of training a transformer from scratch in MLX
- Prototyping small language models on Apple Silicon
- Baseline for TinyStories generation experiments

**Not suitable for:** production use, factual queries, long-form writing, or any application requiring reliability.

## Citation

If you use this model or code, please cite:

```bibtex
@misc{nano-gpt-mlx,
  title={nano-gpt-mlx: A minimal GPT-2 in pure MLX},
  author={Atharva},
  year={2025},
  howpublished={\url{https://github.com/Atharva2099/nano-gpt-mlx}}
}
```

## Acknowledgments

- [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset by Ronen Eldan
- [MLX](https://github.com/ml-explore/mlx) by Apple
- GPT-2 architecture by OpenAI
