import mlx.core as mx
import mlx.nn as nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, context_length):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

        # Causal mask: (1, 1, context_length, context_length)
        self.mask = mx.tril(mx.ones((context_length, context_length)))[None, None, :, :]

    def __call__(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, n_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale  # (B, n_heads, T, T)
        scores = mx.where(self.mask[:, :, :T, :T] == 0, float("-inf"), scores)
        attn = mx.softmax(scores, axis=-1)

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, mlp_dim, context_length):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, context_length)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, d_model),
        )

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class NanoGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, mlp_dim, context_length):
        super().__init__()
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(context_length, d_model)
        self.blocks = [TransformerBlock(d_model, n_heads, mlp_dim, context_length) for _ in range(n_layers)]
        self.ln_f = nn.LayerNorm(d_model)

    def __call__(self, x):
        B, T = x.shape
        pos = mx.arange(T)
        x = self.token_embed(x) + self.pos_embed(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        # Tied embeddings: reuse token_embed weight as output projection
        logits = x @ self.token_embed.weight.T
        return logits

    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
        repetition_penalty=1.0,
    ):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.context_length:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if repetition_penalty != 1.0:
                vocab_idx = mx.arange(logits.shape[-1], dtype=mx.int32)
                adjusted_rows = []
                for b in range(logits.shape[0]):
                    seen = mx.array(list(set(idx[b].tolist())), dtype=mx.int32)
                    row_mask = mx.any(mx.expand_dims(seen, 1) == mx.expand_dims(vocab_idx, 0), axis=0)
                    row_logits = mx.where(row_mask, logits[b] / repetition_penalty, logits[b])
                    adjusted_rows.append(row_logits)
                logits = mx.stack(adjusted_rows, axis=0)
            if top_k is not None:
                k = min(top_k, logits.shape[-1])
                kth = mx.topk(logits, k, axis=-1)[:, -1:]
                logits = mx.where(logits < kth, float("-inf"), logits)
            next_token = mx.random.categorical(logits, axis=-1)
            idx = mx.concatenate([idx, next_token[:, None]], axis=1)
        return idx
