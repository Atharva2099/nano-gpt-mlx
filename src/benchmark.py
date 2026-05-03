import os
import math
import mlx.core as mx
from src import config
from src.data import vocab_size, get_batch
from src.model import NanoGPT
from src.train import cross_entropy_loss


def estimate_split_loss(model, split, eval_iters, batch_size, context_length):
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(split, batch_size, context_length)
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        losses.append(loss.item())
    return sum(losses) / len(losses)


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
checkpoint_name = os.environ.get("CKPT", "best.npz")
ckpt = os.path.join(checkpoint_dir, checkpoint_name)
if not os.path.exists(ckpt):
    raise FileNotFoundError(f"Missing checkpoint: {ckpt}")

model.load_weights(ckpt)
model.eval()

train_loss = estimate_split_loss(
    model, "train", config.eval_iters, config.batch_size, config.context_length
)
val_loss = estimate_split_loss(
    model, "val", config.eval_iters, config.batch_size, config.context_length
)

print(f"Checkpoint: {ckpt}")
print(f"Train loss: {train_loss:.6f}")
print(f"Val loss:   {val_loss:.6f}")
print(f"Train ppl:  {math.exp(train_loss):.6f}")
print(f"Val ppl:    {math.exp(val_loss):.6f}")
