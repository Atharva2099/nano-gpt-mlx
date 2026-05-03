import argparse
import json
import os

from datasets import load_dataset


def pick_first(row, keys):
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def to_record(row):
    # TinyStoriesInstruct variants may expose slightly different field names.
    prompt = pick_first(
        row,
        ["instruction", "prompt", "question", "input", "summary", "features", "words"],
    )
    response = pick_first(
        row,
        ["output", "story", "response", "completion", "text", "answer"],
    )

    # Fallback for single-text-field datasets:
    # if we only have a response-like string, use a generic instruction prompt.
    if not prompt and response:
        prompt = "Write a short story."

    # Last-resort fallback: pick longest available string as response and keep generic prompt.
    if not response:
        string_vals = [v.strip() for v in row.values() if isinstance(v, str) and v.strip()]
        if string_vals:
            response = max(string_vals, key=len)
            if not prompt:
                prompt = "Write a short story."

    return {"prompt": prompt, "response": response}


def filter_valid(records):
    out = []
    for r in records:
        if r["prompt"] and r["response"]:
            out.append(r)
    return out


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stream_write_split(ds_split, out_path, split_name, log_every=500000):
    written = 0
    skipped = 0
    with open(out_path, "w") as f:
        for i, row in enumerate(ds_split):
            rec = to_record(row)
            if rec["prompt"] and rec["response"]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1
            else:
                skipped += 1
            if (i + 1) % log_every == 0:
                print(
                    f"[{split_name}] processed={i + 1:,} written={written:,} skipped={skipped:,}"
                )
    print(f"[{split_name}] done processed={i + 1:,} written={written:,} skipped={skipped:,}")
    return written, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Prepare TinyStoriesInstruct SFT data with train/test/holdout_test splits."
    )
    parser.add_argument("--out-dir", default="data/tinystories_instruct")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--holdout-ratio", type=float, default=0.05)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap total dataset size before splitting (default: use full dataset).",
    )
    args = parser.parse_args()

    ratios_sum = args.train_ratio + args.test_ratio + args.holdout_ratio
    if abs(ratios_sum - 1.0) > 1e-6:
        raise ValueError("train/test/holdout ratios must sum to 1.0")

    os.makedirs(args.out_dir, exist_ok=True)

    ds = load_dataset("roneneldan/TinyStoriesInstruct")
    if "train" not in ds:
        raise ValueError("Expected a train split in roneneldan/TinyStoriesInstruct")
    train_ds = ds["train"]
    print(f"Dataset columns: {train_ds.column_names}")
    print(f"Full dataset size: {len(train_ds):,}")

    if args.max_samples is not None and args.max_samples < len(train_ds):
        train_ds = train_ds.shuffle(seed=args.seed).select(range(args.max_samples))
        print(f"Using capped subset: {args.max_samples:,} samples")

    first_split = train_ds.train_test_split(
        test_size=(1.0 - args.train_ratio), seed=args.seed, shuffle=True
    )
    train_split = first_split["train"]
    rem_split = first_split["test"]

    rem_total = args.test_ratio + args.holdout_ratio
    holdout_frac_in_rem = args.holdout_ratio / rem_total
    second_split = rem_split.train_test_split(
        test_size=holdout_frac_in_rem, seed=args.seed + 1, shuffle=True
    )
    test_split = second_split["train"]
    holdout_split = second_split["test"]

    train_path = os.path.join(args.out_dir, "train.jsonl")
    test_path = os.path.join(args.out_dir, "test.jsonl")
    holdout_path = os.path.join(args.out_dir, "holdout_test.jsonl")

    train_written, _ = stream_write_split(train_split, train_path, "train")
    test_written, _ = stream_write_split(test_split, test_path, "test")
    holdout_written, _ = stream_write_split(holdout_split, holdout_path, "holdout_test")

    print(f"Saved: {train_path} ({train_written} rows)")
    print(f"Saved: {test_path} ({test_written} rows)")
    print(f"Saved: {holdout_path} ({holdout_written} rows)")


if __name__ == "__main__":
    main()
