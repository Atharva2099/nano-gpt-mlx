import os
import mlx.core as mx
from src import config
from src.data import vocab_size, encode, decode
from src.model import NanoGPT


mx.random.seed(config.seed)

model = NanoGPT(
    vocab_size,
    config.d_model,
    config.n_heads,
    config.n_layers,
    config.mlp_dim,
    config.context_length,
)

checkpoint_dir = getattr(config, "checkpoint_dir", "checkpoints")
best_ckpt = os.path.join(checkpoint_dir, "best.npz")
if not os.path.exists(best_ckpt):
    raise FileNotFoundError(f"Missing checkpoint: {best_ckpt}")

model.load_weights(best_ckpt)
model.eval()

sample_temperature = getattr(config, "sample_temperature", 0.8)
sample_top_k = getattr(config, "sample_top_k", 20)
sample_repetition_penalty = getattr(config, "sample_repetition_penalty", 1.0)
sample_count = getattr(config, "sample_count", 3)
sample_max_new_tokens = getattr(config, "sample_max_new_tokens", 200)
sample_prompt = getattr(config, "sample_prompt", "Once upon a time")
sample_prompts = getattr(config, "sample_prompts", [sample_prompt])
if not sample_prompts:
    sample_prompts = [sample_prompt]

print(f"Loaded best checkpoint: {best_ckpt}")
for i in range(sample_count):
    prompt = sample_prompts[i % len(sample_prompts)]
    context = mx.array([encode(prompt)], dtype=mx.int32)
    generated = model.generate(
        context,
        max_new_tokens=sample_max_new_tokens,
        temperature=sample_temperature,
        top_k=sample_top_k,
        repetition_penalty=sample_repetition_penalty,
    )[0].tolist()
    print(f"--- Eval Sample {i + 1} | Prompt: {prompt} ---\n{decode(generated)}\n-----------------")
