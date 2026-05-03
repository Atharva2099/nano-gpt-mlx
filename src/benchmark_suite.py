import argparse
import json
import math
import os
import time
from datetime import datetime

import mlx.core as mx

from src import config
from src.data import vocab_size, get_batch, encode, decode
from src.model import NanoGPT
from src.train import cross_entropy_loss
from mlx.utils import tree_flatten


def count_params(model):
    return sum(p.size for _, p in tree_flatten(model.parameters()))


def eval_metrics(model, split, eval_iters, batch_size, context_length):
    losses = []
    correct = 0
    total = 0
    total_tokens = 0
    total_time = 0.0

    for _ in range(eval_iters):
        x, y = get_batch(split, batch_size, context_length)
        t0 = time.time()
        logits = model(x)
        loss = cross_entropy_loss(logits, y)
        mx.eval(loss, logits)
        t1 = time.time()

        losses.append(loss.item())
        pred = mx.argmax(logits, axis=-1)
        matches = (pred == y).astype(mx.int32).sum().item()
        correct += int(matches)
        batch_tokens = x.shape[0] * x.shape[1]
        total += batch_tokens
        total_tokens += batch_tokens
        total_time += (t1 - t0)

    mean_loss = sum(losses) / len(losses)
    return {
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss),
        "bpc": mean_loss / math.log(2.0),
        "top1_accuracy": correct / total if total else 0.0,
        "tokens_per_sec": total_tokens / total_time if total_time > 0 else 0.0,
    }


def generate_samples(model, prompts, sample_count):
    sample_temperature = getattr(config, "sample_temperature", 0.8)
    sample_top_k = getattr(config, "sample_top_k", 20)
    sample_repetition_penalty = getattr(config, "sample_repetition_penalty", 1.0)
    sample_max_new_tokens = getattr(config, "sample_max_new_tokens", 200)

    out = []
    for i in range(sample_count):
        prompt = prompts[i % len(prompts)]
        context = mx.array([encode(prompt)], dtype=mx.int32)
        generated = model.generate(
            context,
            max_new_tokens=sample_max_new_tokens,
            temperature=sample_temperature,
            top_k=sample_top_k,
            repetition_penalty=sample_repetition_penalty,
        )[0].tolist()
        out.append({"prompt": prompt, "text": decode(generated)})
    return out


def build_model(cfg):
    return NanoGPT(
        vocab_size,
        cfg.d_model,
        cfg.n_heads,
        cfg.n_layers,
        cfg.mlp_dim,
        cfg.context_length,
    )


def run_one(args, ckpt_name, seed_override=None, d_model_override=None, context_override=None):
    class Cfg:
        pass

    cfg = Cfg()
    for k, v in config.__dict__.items():
        if not k.startswith("__"):
            setattr(cfg, k, v)

    if seed_override is not None:
        cfg.seed = seed_override
    if d_model_override is not None:
        cfg.d_model = d_model_override
    if context_override is not None:
        cfg.context_length = context_override

    mx.random.seed(cfg.seed)
    model = build_model(cfg)

    ckpt_path = os.path.join(cfg.checkpoint_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    model.load_weights(ckpt_path)
    model.eval()

    eval_iters = args.eval_iters if args.eval_iters is not None else cfg.eval_iters
    batch_size = args.batch_size if args.batch_size is not None else cfg.batch_size
    context_length = cfg.context_length

    train_stats = eval_metrics(model, "train", eval_iters, batch_size, context_length)
    val_stats = eval_metrics(model, "val", eval_iters, batch_size, context_length)

    prompts = getattr(cfg, "sample_prompts", ["Once upon a time"])
    samples = generate_samples(model, prompts, args.sample_count)

    return {
        "checkpoint": ckpt_path,
        "seed": cfg.seed,
        "params": count_params(model),
        "d_model": cfg.d_model,
        "context_length": cfg.context_length,
        "eval_iters": eval_iters,
        "batch_size": batch_size,
        "train": train_stats,
        "val": val_stats,
        "samples": samples,
    }


def parse_csv_ints(value):
    if not value:
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Benchmark suite logger for MLX char-level LM.")
    parser.add_argument("--ckpt", default="latest.npz", help="Checkpoint filename inside checkpoints/.")
    parser.add_argument("--eval-iters", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--sample-count", type=int, default=3)
    parser.add_argument("--seeds", default="", help="Comma-separated seed sweep, e.g. 1337,42,7")
    parser.add_argument("--d-models", default="", help="Comma-separated d_model sweep, e.g. 96,128")
    parser.add_argument("--contexts", default="", help="Comma-separated context sweep, e.g. 128,192")
    parser.add_argument("--tag", default="char_baseline")
    args = parser.parse_args()

    seeds = parse_csv_ints(args.seeds) or [None]
    d_models = parse_csv_ints(args.d_models) or [None]
    contexts = parse_csv_ints(args.contexts) or [None]

    runs = []
    for s in seeds:
        for d in d_models:
            for c in contexts:
                runs.append(run_one(args, args.ckpt, seed_override=s, d_model_override=d, context_override=c))

    val_losses = [r["val"]["loss"] for r in runs]
    val_ppls = [r["val"]["perplexity"] for r in runs]

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tag": args.tag,
        "checkpoint_name": args.ckpt,
        "num_runs": len(runs),
        "val_loss_mean": sum(val_losses) / len(val_losses),
        "val_loss_std": math.sqrt(sum((x - (sum(val_losses) / len(val_losses))) ** 2 for x in val_losses) / len(val_losses)),
        "val_ppl_mean": sum(val_ppls) / len(val_ppls),
        "val_ppl_std": math.sqrt(sum((x - (sum(val_ppls) / len(val_ppls))) ** 2 for x in val_ppls) / len(val_ppls)),
        "runs": runs,
    }

    out_dir = os.path.join("docs", "benchmarks")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.tag}.json")
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved benchmark log: {out_file}")
    print(f"val_loss_mean={summary['val_loss_mean']:.6f}, val_ppl_mean={summary['val_ppl_mean']:.6f}")


if __name__ == "__main__":
    main()
