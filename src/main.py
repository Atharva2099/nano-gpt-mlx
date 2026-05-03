import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
import os
from src import config
from src.data import vocab_size, get_batch, encode, decode
from src.model import NanoGPT
from src.train import train

def count_params(model):
    return sum(p.size for _, p in tree_flatten(model.parameters()))

mx.random.seed(config.seed)

model = NanoGPT(vocab_size, config.d_model, config.n_heads, config.n_layers, config.mlp_dim, config.context_length)
checkpoint_dir = getattr(config, "checkpoint_dir", "checkpoints")
latest_ckpt = os.path.join(checkpoint_dir, "latest.npz")
latest_step_file = os.path.join(checkpoint_dir, "latest_step.txt")
latest_optim_ckpt = os.path.join(checkpoint_dir, "latest_optim.npz")
start_iter = 0
optimizer_state = None

if os.path.exists(latest_ckpt):
    model.load_weights(latest_ckpt)
    if os.path.exists(latest_step_file):
        with open(latest_step_file) as f:
            last_step = int(f.read().strip())
        start_iter = last_step + 1
        print(f"Loaded checkpoint from step {last_step}: {latest_ckpt}")
    else:
        print(f"Loaded checkpoint: {latest_ckpt}")
    if os.path.exists(latest_optim_ckpt):
        optimizer_state = tree_unflatten(mx.load(latest_optim_ckpt))
        print(f"Loaded optimizer state: {latest_optim_ckpt}")
else:
    print("No checkpoint found. Starting fresh.")

model.eval()

n_params = count_params(model)
print(f"Model params: {n_params / 1e6:.3f}M")

# Quick sanity check: forward pass
x, y = get_batch("train", config.batch_size, config.context_length)
logits = model(x)
print(f"Logits shape: {logits.shape}")

model.train()
if start_iter >= config.max_iters:
    print(
        f"Checkpoint is already at step {start_iter - 1}, "
        f"which reaches/exceeds max_iters={config.max_iters}. "
        "Increase max_iters in src/config.py to continue training."
    )
    raise SystemExit(0)

train(
    model,
    get_batch,
    config,
    encode,
    decode,
    start_iter=start_iter,
    optimizer_state=optimizer_state,
)
