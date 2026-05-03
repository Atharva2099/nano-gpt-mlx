import json
import mlx.core as mx
from src import config
from src.data import encode

TRAIN_PATH = "data/tinystories_instruct/train.jsonl"
TEST_PATH = "data/tinystories_instruct/test.jsonl"


def load_records(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


train_records = load_records(TRAIN_PATH)
test_records = load_records(TEST_PATH)


def tokenize_record(rec, context_length):
    prompt_text = rec["prompt"] + "\n\n"
    full_text = prompt_text + rec["response"]

    tokens = encode(full_text)
    prompt_tokens = encode(prompt_text)

    # Truncate if needed
    if len(tokens) > context_length:
        tokens = tokens[:context_length]

    # Targets: shift by 1, pad last position
    targets = tokens[1:] + [0]

    # Loss mask: 1 for response tokens, 0 for prompt tokens.
    # At position i we predict tokens[i+1].
    # Include positions where i+1 >= len(prompt_tokens), i.e. i >= len(prompt_tokens) - 1.
    mask = [0] * len(tokens)
    start = max(0, len(prompt_tokens) - 1)
    for i in range(start, len(tokens) - 1):
        mask[i] = 1

    # Pad to context_length
    pad_len = context_length - len(tokens)
    if pad_len > 0:
        tokens = tokens + [0] * pad_len
        targets = targets + [0] * pad_len
        mask = mask + [0] * pad_len
    else:
        tokens = tokens[:context_length]
        targets = targets[:context_length]
        mask = mask[:context_length]

    return tokens, targets, mask


def prepare_all(records, context_length):
    all_tokens = []
    all_targets = []
    all_masks = []
    for rec in records:
        t, tg, m = tokenize_record(rec, context_length)
        all_tokens.append(t)
        all_targets.append(tg)
        all_masks.append(m)
    return (
        mx.array(all_tokens, dtype=mx.int32),
        mx.array(all_targets, dtype=mx.int32),
        mx.array(all_masks, dtype=mx.float32),
    )


train_x, train_y, train_mask = prepare_all(train_records, config.context_length)
test_x, test_y, test_mask = prepare_all(test_records, config.context_length)


def get_batch(split, batch_size):
    x = train_x if split == "train" else test_x
    y = train_y if split == "train" else test_y
    m = train_mask if split == "train" else test_mask
    n = x.shape[0]
    ix = mx.random.randint(0, n, (batch_size,))
    bx = mx.stack([x[int(i)] for i in ix])
    by = mx.stack([y[int(i)] for i in ix])
    bm = mx.stack([m[int(i)] for i in ix])
    return bx, by, bm
