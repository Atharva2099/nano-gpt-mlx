# Tokenizer Versions

This directory contains BPE tokenizers trained on the TinyStories dataset at different vocabulary sizes. All were trained with `src/train_tokenizer.py` using ByteLevel pre-tokenization.

## Current (Recommended)

| Directory | Vocab Size | Status | Notes |
|-----------|-----------|--------|-------|
| `tokenizer_bpe4096/` | 4096 | **Active** | Used for current pretraining + SFT. Best quality. |

## Historical Versions

| Directory | Vocab Size | Status | Notes |
|-----------|-----------|--------|-------|
| `tokenizer_bpe1024_v1/` | 1024 | Archived | First BPE attempt. Too small — caused subword loops and gibberish. |
| `tokenizer_bpe256_v1/` | 256 | Archived | Even smaller vocab, largely experimental. |

The `_v1` suffix denotes the first generation of tokenizers. Future iterations (if any) would use `_v2`, etc.

## Size Comparison

| Vocab | File Size | Avg Tokens / Story | Quality |
|-------|-----------|-------------------|---------|
| 256 | ~70 KB | Very high | Poor — many splits per word |
| 1024 | ~270 KB | High | Fair — subword repetition issues |
| **4096** | **~1.1 MB** | **Moderate** | **Best — clean word boundaries** |

## Usage

Set the desired tokenizer in `src/config.py`:

```python
tokenizer_type = "bpe"
bpe_tokenizer_path = "artifacts/tokenizer_bpe4096/tokenizer.json"
```

## Training

To train a new tokenizer:

```bash
uv run python src/train_tokenizer.py \
  --train-file data/train.txt \
  --out-dir artifacts/tokenizer_bpe4096 \
  --vocab-size 4096
```
