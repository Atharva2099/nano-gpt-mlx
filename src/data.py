import os
import tarfile
import urllib.request
import json
import mlx.core as mx
from src import config

try:
    from tokenizers import Tokenizer
except ImportError:
    Tokenizer = None

DATA_DIR = "data"
TAR_PATH = os.path.join(DATA_DIR, "TinyStories_all_data.tar.gz")
TAR_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
VAL_FILE = os.path.join(DATA_DIR, "val.txt")
MAX_TRAIN_STORIES = 20000
MAX_VAL_STORIES = 2000


def download():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(TAR_PATH):
        print("Downloading TinyStories dataset (~1.6GB, one-time)...")
        urllib.request.urlretrieve(TAR_URL, TAR_PATH)
        print("Download complete.")


def extract_subset():
    if os.path.exists(TRAIN_FILE) and os.path.exists(VAL_FILE):
        return

    print("Extracting subset of stories...")
    train_texts = []
    val_texts = []

    def add_text(text):
        if len(train_texts) < MAX_TRAIN_STORIES:
            train_texts.append(text)
        elif len(val_texts) < MAX_VAL_STORIES:
            val_texts.append(text)

    with tarfile.open(TAR_PATH, "r:gz") as tar:
        members = tar.getmembers()

        # Try .txt files first
        for m in members:
            if not m.isfile() or not m.name.endswith(".txt"):
                continue
            f = tar.extractfile(m)
            if not f:
                continue

            # Read first 50MB to avoid memory issues
            chunk = f.read(50 * 1024 * 1024).decode("utf-8")
            stories = [s.strip() for s in chunk.split("\n\n") if s.strip()]

            if "train" in m.name.lower():
                train_texts.extend(stories[:MAX_TRAIN_STORIES])
            elif "val" in m.name.lower() or "valid" in m.name.lower():
                val_texts.extend(stories[:MAX_VAL_STORIES])

        # Try .json / .jsonl
        if not train_texts:
            for m in members:
                if not m.isfile() or not (m.name.endswith(".json") or m.name.endswith(".jsonl")):
                    continue
                f = tar.extractfile(m)
                if not f:
                    continue

                content = f.read().decode("utf-8")

                # Try as single JSON
                try:
                    data = json.loads(content)
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, str) and item.strip():
                                add_text(item.strip())
                            elif isinstance(item, dict):
                                t = item.get("text", item.get("story", ""))
                                if t:
                                    add_text(t)
                    elif isinstance(data, dict):
                        t = data.get("text", data.get("story", ""))
                        if t:
                            add_text(t)
                except json.JSONDecodeError:
                    # Line by line
                    for line in content.split("\n"):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            if isinstance(data, list):
                                for item in data:
                                    if isinstance(item, str) and item.strip():
                                        add_text(item.strip())
                                    elif isinstance(item, dict):
                                        t = item.get("text", item.get("story", ""))
                                        if t:
                                            add_text(t)
                            elif isinstance(data, dict):
                                t = data.get("text", data.get("story", ""))
                                if t:
                                    add_text(t)
                        except json.JSONDecodeError:
                            continue

                if len(train_texts) >= MAX_TRAIN_STORIES and len(val_texts) >= MAX_VAL_STORIES:
                    break

    with open(TRAIN_FILE, "w") as f:
        f.write("\n\n".join(train_texts))
    with open(VAL_FILE, "w") as f:
        f.write("\n\n".join(val_texts))

    print(f"Saved {len(train_texts)} train stories and {len(val_texts)} val stories.")


download()
extract_subset()

with open(TRAIN_FILE) as f:
    train_text = f.read()
with open(VAL_FILE) as f:
    val_text = f.read()

if getattr(config, "tokenizer_type", "char") == "bpe":
    if Tokenizer is None:
        raise ImportError("tokenizers package is required for BPE mode. Install tokenizers and retry.")
    tokenizer_path = getattr(config, "bpe_tokenizer_path", "artifacts/tokenizer_bpe/tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(f"Missing BPE tokenizer file: {tokenizer_path}")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()

    def encode(s):
        return tokenizer.encode(s).ids

    def decode(ids):
        return tokenizer.decode(ids)

    train_ids = mx.array(encode(train_text), dtype=mx.int32)
    val_ids = mx.array(encode(val_text), dtype=mx.int32)
else:
    # Char-level vocab from train set
    vocab = sorted(set(train_text))
    vocab_size = len(vocab)

    stoi = {c: i for i, c in enumerate(vocab)}
    itos = {i: c for i, c in enumerate(vocab)}

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])

    train_ids = mx.array(encode(train_text), dtype=mx.int32)
    val_ids = mx.array(encode(val_text), dtype=mx.int32)


def get_batch(split, batch_size, context_length):
    data = train_ids if split == "train" else val_ids
    ix = mx.random.randint(0, len(data) - context_length, (batch_size,))
    x = mx.stack([data[int(i) : int(i) + context_length] for i in ix])
    y = mx.stack([data[int(i) + 1 : int(i) + context_length + 1] for i in ix])
    return x, y
