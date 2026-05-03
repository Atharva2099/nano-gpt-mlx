import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import os
import time


def sft_loss(logits, targets, mask):
    losses = nn.losses.cross_entropy(logits, targets, reduction="none")
    masked = losses * mask
    return mx.sum(masked) / mx.maximum(mx.sum(mask), 1.0)


def estimate_loss(model, get_batch, eval_iters, batch_size):
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(eval_iters):
            x, y, m = get_batch(split, batch_size)
            logits = model(x)
            loss = sft_loss(logits, y, m)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    return out


def train(model, get_batch, config, encode, decode, start_iter=0, optimizer_state=None):
    optimizer = optim.AdamW(learning_rate=config.sft_learning_rate)
    if optimizer_state is not None:
        optimizer.state = optimizer_state
    checkpoint_dir = getattr(config, "sft_checkpoint_dir", "checkpoints_sft")
    sample_temperature = getattr(config, "sample_temperature", 0.8)
    sample_top_k = getattr(config, "sample_top_k", 20)
    sample_repetition_penalty = getattr(config, "sample_repetition_penalty", 1.0)
    sample_count = getattr(config, "sample_count", 3)
    sample_max_new_tokens = getattr(config, "sample_max_new_tokens", 200)
    sample_prompt = getattr(config, "sft_sample_prompt", "Write a short story.")
    sample_prompts = getattr(config, "sft_sample_prompts", [sample_prompt])
    if not sample_prompts:
        sample_prompts = [sample_prompt]
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val = float("inf")
    best_val_path = os.path.join(checkpoint_dir, "best_val.txt")
    if os.path.exists(best_val_path):
        with open(best_val_path) as f:
            best_val = float(f.read().strip())

    def save_checkpoint(iteration, is_best):
        latest_path = os.path.join(checkpoint_dir, "latest.npz")
        best_path = os.path.join(checkpoint_dir, "best.npz")
        step_path = os.path.join(checkpoint_dir, "latest_step.txt")
        latest_optim_path = os.path.join(checkpoint_dir, "latest_optim.npz")

        model.save_weights(latest_path)
        mx.savez(latest_optim_path, **tree_flatten(optimizer.state, destination={}))
        with open(step_path, "w") as f:
            f.write(str(iteration))
        with open(best_val_path, "w") as f:
            f.write(f"{best_val:.6f}")

        if is_best:
            model.save_weights(best_path)

    def step(x, y, m):
        def loss_fn(model):
            logits = model(x)
            return sft_loss(logits, y, m)

        loss, grads = mx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)
        return loss

    print(f"SFT training from step {start_iter} to {config.sft_max_iters - 1}...")
    for iter in range(start_iter, config.sft_max_iters):
        t0 = time.time()
        x, y, m = get_batch("train", config.sft_batch_size)
        loss = step(x, y, m)
        mx.eval(loss, model.parameters())
        t1 = time.time()

        if iter % config.sft_eval_interval == 0 or iter == config.sft_max_iters - 1:
            losses = estimate_loss(model, get_batch, config.sft_eval_iters, config.sft_batch_size)
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, dt {(t1 - t0) * 1000:.2f}ms"
            )
            is_best = losses["val"] < best_val
            if is_best:
                best_val = losses["val"]
            save_checkpoint(iter, is_best)

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
                print(f"--- Sample {i + 1} | Prompt: {prompt} ---\n{decode(generated)}\n-----------------")

    return model
