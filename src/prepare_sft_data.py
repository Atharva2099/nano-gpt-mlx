import argparse
import json
import os

from datasets import load_dataset


def parse_stories(ds_split):
    """
    TinyStoriesInstruct is a flat sequence of text lines.
    Each story block looks like:

        Features: Dialogue
        Words: quit, oak, gloomy
        Summary: ...
        Story:
        [blank]
        Once upon a time...
        ...
        <|endoftext|>

    We group rows between <|endoftext|> markers into single (prompt, response) pairs.
    """
    stories = []
    current_lines = []

    for row in ds_split:
        line = row.get("text", "").strip()
        if line == "<|endoftext|>":
            if current_lines:
                parsed = _build_record(current_lines)
                if parsed:
                    stories.append(parsed)
                current_lines = []
        else:
            current_lines.append(line)

    # trailing block without <|endoftext|>
    if current_lines:
        parsed = _build_record(current_lines)
        if parsed:
            stories.append(parsed)

    return stories


def _build_record(lines):
    """Turn a block of lines into {"prompt": ..., "response": ...}."""
    if not lines:
        return None

    # Collect constraints from header lines
    constraints = []
    story_lines = []
    in_story = False

    for line in lines:
        lower = line.lower()
        if lower.startswith("features:"):
            constraints.append(("features", line[9:].strip()))
        elif lower.startswith("words:"):
            constraints.append(("words", line[6:].strip()))
        elif lower.startswith("summary:"):
            constraints.append(("summary", line[8:].strip()))
        elif lower.startswith("story:"):
            in_story = True
            remainder = line[6:].strip()
            if remainder:
                story_lines.append(remainder)
        elif in_story and line:
            story_lines.append(line)
        elif not in_story and line:
            # stray line before Story: — ignore
            pass

    if not story_lines:
        return None

    story_text = " ".join(story_lines)
    if len(story_text) < 40:
        return None

    # Build a diverse prompt from constraints
    prompt_parts = []
    for ctype, cval in constraints:
        if ctype == "features":
            prompt_parts.append(f"with features: {cval}")
        elif ctype == "words":
            prompt_parts.append(f"using the words: {cval}")
        elif ctype == "summary":
            prompt_parts.append(f"about: {cval}")

    if prompt_parts:
        prompt = "Write a short story " + ", ".join(prompt_parts) + "."
    else:
        prompt = "Write a short story."

    return {"prompt": prompt, "response": story_text}


def stream_write_records(records, out_path, split_name, log_every=50000):
    written = 0
    skipped = 0
    with open(out_path, "w") as f:
        for i, rec in enumerate(records):
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
        description="Prepare TinyStoriesInstruct SFT data with real prompt diversity."
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
        help="Cap total stories before splitting (default: use full dataset).",
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
    print(f"Raw rows: {len(train_ds):,}")

    print("Parsing story blocks...")
    records = parse_stories(train_ds)
    print(f"Extracted {len(records):,} stories")

    # Quick diversity check
    unique_prompts = set(r["prompt"] for r in records)
    print(f"Unique prompts: {len(unique_prompts):,}")

    if args.max_samples is not None and args.max_samples < len(records):
        import random
        random.seed(args.seed)
        records = random.sample(records, args.max_samples)
        print(f"Using capped subset: {args.max_samples:,} stories")

    first_split_idx = int(len(records) * args.train_ratio)
    train_records = records[:first_split_idx]
    rem_records = records[first_split_idx:]

    rem_total = args.test_ratio + args.holdout_ratio
    holdout_frac = args.holdout_ratio / rem_total
    second_split_idx = int(len(rem_records) * (1 - holdout_frac))
    test_records = rem_records[:second_split_idx]
    holdout_records = rem_records[second_split_idx:]

    train_path = os.path.join(args.out_dir, "train.jsonl")
    test_path = os.path.join(args.out_dir, "test.jsonl")
    holdout_path = os.path.join(args.out_dir, "holdout_test.jsonl")

    train_written, _ = stream_write_records(train_records, train_path, "train")
    test_written, _ = stream_write_records(test_records, test_path, "test")
    holdout_written, _ = stream_write_records(holdout_records, holdout_path, "holdout_test")

    print(f"Saved: {train_path} ({train_written} rows)")
    print(f"Saved: {test_path} ({test_written} rows)")
    print(f"Saved: {holdout_path} ({holdout_written} rows)")


if __name__ == "__main__":
    main()
