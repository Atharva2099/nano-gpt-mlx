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

A 3.5M parameter GPT-2 style decoder-only language model trained from scratch in pure MLX on Apple Silicon.

> **Note on naming:** The repo name reflects the original 0.6M parameter release. The current model has been scaled to **3.5M parameters** with 256-token context and a 4096-token BPE vocabulary. See the Model Evolution section below for the full history.

## Model Description

This is a minimal transformer language model designed to run and train on an M1 Mac with 8GB RAM. It was trained in two stages:

1. **Pretraining** on TinyStories (next-token prediction)
2. **Supervised Fine-Tuning** on TinyStoriesInstruct (instruction-following with prompt masking and diverse prompts)

All training code is implemented in MLX only — no PyTorch, TensorFlow, or JAX.

### Current Architecture (v4)

| Spec | Value |
|------|-------|
| Parameters | ~3.5M |
| Layers | 6 |
| d_model | 192 |
| Attention heads | 6 |
| MLP dimension | 768 |
| Context length | 256 |
| Vocabulary | 4096 (BPE) |
| Embeddings | Tied (input/output) |

## Training

### Pretraining
- **Dataset**: [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)
- **Objective**: Causal language modeling (next-token prediction)
- **Steps**: 16,000
- **Learning rate**: 1e-3
- **Batch size**: 16
- **Optimizer**: AdamW
- **Final val loss**: 2.00

### Supervised Fine-Tuning
- **Dataset**: [roneneldan/TinyStoriesInstruct](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct) (300k subset)
- **Objective**: Causal LM with prompt masking (loss only on response tokens)
- **Steps**: 10,000
- **Learning rate**: 1e-4
- **Batch size**: 8
- **Optimizer**: AdamW
- **Final val loss**: 1.82
- **Prompt diversity**: ~2.4M unique prompts extracted from story constraints (words, features, summaries)

## Model Evolution

| Version | Tokenizer | Params | d_model | Layers | Context | Pretrain Loss | SFT Loss | Status |
|---------|-----------|--------|---------|--------|---------|---------------|----------|--------|
| **v4 (current)** | BPE 4096 | **3.5M** | 192 | 6 | 256 | 2.00 | **1.82** | Active |
| v3 | BPE 4096 | 0.95M | 96 | 4 | 128 | 2.16 | 2.20 | Archived |
| v2 | BPE 1024 | 0.62M | 96 | 4 | 128 | 2.00 | 2.01 | Archived |
| v1.5 | BPE 256 | 0.47M | 96 | 4 | 128 | 2.24 | — | Archived |
| v1 | Char-level | 0.47M | 96 | 4 | 128 | 0.77 | — | Archived |

### Key Iterations

- **v1 (char-level):** Baseline with character vocabulary (~65 chars). Trained as proof-of-concept.
- **v1.5→v2 (BPE scaling):** Moved to BPE tokenization. 256 vocab was too small; 1024 vocab produced real words but caused subword loops ("machones", "sungony").
- **v3 (4096 vocab, same model):** Larger vocab cleaned up word generation, but the 0.6M parameter model still couldn't maintain coherence.
- **v4 (3.5M params, 256 ctx):** Scaled d_model to 192 and layers to 6. This was the biggest quality jump — repetition loops disappeared, stories gained plot structure, and characters persisted across sentences.
- **SFT fix:** The initial SFT script treated every row as an independent example and defaulted all prompts to `"Write a short story."`. Fixing `prepare_sft_data.py` to group multi-line story blocks and extract real constraints (words, features, summaries) produced ~2.4M unique prompts and actual instruction-following behavior.

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
prompt = "Write a short story using the words: brave, mountain, treasure."
ids = tokenizer.encode(prompt).ids
context = mx.array([ids], dtype=mx.int32)
output = model.generate(
    context,
    max_new_tokens=100,
    temperature=0.8,
    top_k=20,
    repetition_penalty=1.3,
)
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

## Files in this Repo

| File | Description |
|------|-------------|
| `model.npz` | SFT model weights (3.5M params, instruction-tuned) |
| `pretrained_model.npz` | Baseline pretrained model (before SFT) |
| `tokenizer/tokenizer.json` | BPE tokenizer (4096 vocab) |

## Evaluation

Final validation cross-entropy loss after training:

| Stage | Dataset | Val Loss | Perplexity |
|-------|---------|----------|------------|
| Pretraining | TinyStories | 2.00 | 7.4 |
| SFT | TinyStoriesInstruct | 1.82 | 6.2 |

The SFT model achieves lower loss on its target domain (instruction-following stories) than the pretrained model does on raw TinyStories, despite the distribution shift. This indicates the SFT stage successfully adapted the model.

### Qualitative Behavior

- **Instruction following:** Responds to specific constraints like word lists, feature tags, and story summaries.
- **Coherence:** Multi-sentence stories with characters, problems, and resolutions.
- **Limitations:** Character names can drift across long generations, tense mixing occurs, and very complex instructions exceed model capacity.

## Limitations

- **Tiny scale:** 3.5M parameters is still very small. Output can have coherence gaps.
- **Short context:** 256 tokens limits generation to brief passages.
- **Narrow domain:** Trained only on children's stories. No general knowledge, reasoning, or factual accuracy.
- **Hardware-constrained:** Designed for 8GB M1 Mac; larger models would improve quality significantly.

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
