import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
import os
from src import config
from src.data import vocab_size, encode, decode
from src.model import NanoGPT
from src.sft_data import get_batch
from src.sft_train import train


def count_params(model):
    return sum(p.size for _, p in tree_flatten(model.parameters()))


mx.random.seed(config.seed)

model = NanoGPT(vocab_size, config.d_model, config.n_heads, config.n_layers, config.mlp_dim, config.context_length)

# Load pre-trained weights
pretrain_ckpt = getattr(config, "pretrain_checkpoint_dir", "checkpoints_bpe1024")
latest_ckpt = os.path.join(pretrain_ckpt, "latest.npz")
best_ckpt = os.path.join(pretrain_ckpt, "best.npz")

ckpt_to_load = None
if os.path.exists(best_ckpt):
    ckpt_to_load = best_ckpt
elif os.path.exists(latest_ckpt):
    ckpt_to_load = latest_ckpt

if ckpt_to_load:
    model.load_weights(ckpt_to_load)
    print(f"Loaded pre-trained weights: {ckpt_to_load}")
else:
    print("WARNING: No pre-trained checkpoint found. Starting from scratch.")

# Load SFT checkpoint if resuming
checkpoint_dir = getattr(config, "sft_checkpoint_dir", "checkpoints_sft")
latest_sft_ckpt = os.path.join(checkpoint_dir, "latest.npz")
latest_sft_step = os.path.join(checkpoint_dir, "latest_step.txt")
latest_sft_optim = os.path.join(checkpoint_dir, "latest_optim.npz")
start_iter = 0
optimizer_state = None

if os.path.exists(latest_sft_ckpt):
    model.load_weights(latest_sft_ckpt)
    if os.path.exists(latest_sft_step):
        with open(latest_sft_step) as f:
            last_step = int(f.read().strip())
        start_iter = last_step + 1
        print(f"Loaded SFT checkpoint from step {last_step}: {latest_sft_ckpt}")
    else:
        print(f"Loaded SFT checkpoint: {latest_sft_ckpt}")
    if os.path.exists(latest_sft_optim):
        optimizer_state = tree_unflatten(mx.load(latest_sft_optim))

model.eval()

n_params = count_params(model)
print(f"Model params: {n_params / 1e6:.3f}M")

# Sanity check
x, y, m = get_batch("train", getattr(config, "sft_batch_size", 16))
logits = model(x)
print(f"Logits shape: {logits.shape}")

model.train()
if start_iter >= getattr(config, "sft_max_iters", 2000):
    print(
        f"SFT checkpoint already at step {start_iter - 1}. "
        "Increase sft_max_iters in src/config.py to continue."
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
