import argparse
import os

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing


def main():
    parser = argparse.ArgumentParser(description="Train a TinyStories BPE tokenizer.")
    parser.add_argument("--train-file", default="data/train.txt")
    parser.add_argument("--out-dir", default="artifacts/tokenizer_bpe")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()

    if not os.path.exists(args.train_file):
        raise FileNotFoundError(f"Missing train file: {args.train_file}")

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"],
    )
    tokenizer.train(files=[args.train_file], trainer=trainer)

    bos_id = tokenizer.token_to_id("[BOS]")
    eos_id = tokenizer.token_to_id("[EOS]")
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        pair="[BOS] $A [EOS] $B:1 [EOS]:1",
        special_tokens=[("[BOS]", bos_id), ("[EOS]", eos_id)],
    )

    tokenizer_json = os.path.join(args.out_dir, "tokenizer.json")
    tokenizer.save(tokenizer_json)

    print(f"Saved tokenizer: {tokenizer_json}")
    print(f"Vocab size: {tokenizer.get_vocab_size()}")
    print(f"BOS id: {bos_id}, EOS id: {eos_id}")


if __name__ == "__main__":
    main()
