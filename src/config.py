vocab_size = 128
d_model = 192
n_heads = 6
n_layers = 6
mlp_dim = 768
context_length = 256

batch_size = 16
learning_rate = 1e-3
max_iters = 16000
eval_interval = 500
eval_iters = 200

seed = 1337

# Derived: approx 3.5M params with vocab=4096

checkpoint_dir = "checkpoints_bpe4096"
tokenizer_type = "bpe"  # "char" or "bpe"
bpe_tokenizer_path = "artifacts/tokenizer_bpe4096/tokenizer.json"

sample_temperature = 0.8
sample_top_k = 20
sample_repetition_penalty = 1.3
sample_count = 3
sample_max_new_tokens = 200
sample_prompt = "Once upon a time"
sample_prompts = [
    "Once upon a time",
    "Lily and Tim went to",
    "The little cat was",
    "In the morning,",
    "He felt sad because",
    "They found a box and",
]

# SFT config
sft_learning_rate = 1e-4
sft_batch_size = 8
sft_max_iters = 10000
sft_eval_interval = 200
sft_eval_iters = 50
sft_checkpoint_dir = "checkpoints_sft_bpe4096"
pretrain_checkpoint_dir = "checkpoints_bpe4096"
sft_sample_prompt = "Write a short story."
sft_sample_prompts = [
    "Write a short story.",
    "Tell me a story about a brave dog.",
    "Once upon a time there was a little girl named Emma.",
]
