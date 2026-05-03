# MLX Nano-GPT

## Project Goal
Build the smallest possible GPT-2 style decoder-only language model in **pure MLX**. The model, layers, forward pass, loss, optimizer step, and training loop must be implemented in MLX only. No PyTorch, TensorFlow, JAX, Flax, or any other deep learning framework in the model or training code.

## Scope
- **Hardware**: Train fully on an M1 Mac with 8GB RAM.
- **Dataset**: Use TinyStories.
- **Tokenizer**: Start with a character-level tokenizer.
- **Model Size**: Keep the model under 0.5M parameters.
- **Architecture**:
  - 4 transformer blocks
  - d_model = 96
  - n_heads = 4
  - mlp_dim = 384
  - context length = 128
  - Tied token embedding and output head if possible

## Design Philosophy
- **Minimal & Understandable**: Code should be readable and concise. Avoid excessive boilerplate, guardrails, or extreme try/catch blocks. This is prototyping.
- **MLX-Native**: Use MLX for tensor ops, autograd, modules/layers, optimizer, training loop, and inference/generation.
- **Helper Libraries**: Allowed only for basic Python utilities, file handling, downloading data, and plotting.
- **Grounded in MLX Documentation**: Assume similarity to other libraries (like PyTorch or Flax) where applicable to make functions intuitive.

## Project Structure
```
mlx-nano-gpt/
└── src/
    ├── config.py   # Hyperparameters & model config
    ├── data.py     # TinyStories download, char tokenizer, batching
    ├── model.py    # Pure MLX transformer architecture
    ├── train.py    # Training/validation loops, generation, logging
    ├── main.py     # Entry point: load data, init model, run training
    └── eval.py     # Best-checkpoint generation-only evaluation
```

## Implementation Notes
- MLX uses lazy evaluation. Force computation with `mx.eval()` at key points during training.
- Tied embeddings: Store one `Embedding` weight and reuse it for the final linear projection.
- Causal mask: Precomputed as a boolean upper-triangle mask in `CausalSelfAttention`.
- Optimizer: Use `mlx.optimizers.AdamW` directly.

## Code Rules
1. No external DL frameworks (PyTorch, TF, JAX, Flax, etc.) in model or training code.
2. Keep code simple and readable.
3. Organize into minimal files.
4. Prioritize correctness and clarity over performance tricks.
5. No unnecessary guardrails or safety checks for prototyping.

## Workflow Rules
1. **One file at a time**: Build, review, and get user approval before moving to the next file.
2. **MLX only**: All model, training, and inference code must use MLX only. No PyTorch, TensorFlow, JAX, or Flax anywhere in model/training code.
3. **User runs terminal commands**: The user will run all terminal commands. Do not execute bash commands that run, install, or test code. Only write code to files.
4. **Always use uv**: Use `uv` for Python package and environment workflows (for example `uv pip install ...`, `uv run ...`) instead of `pip` directly.
