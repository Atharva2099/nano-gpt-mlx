# Learnings Log

## 2026-05-01: Logits vs Probabilities in Sampling

`mlx.random.categorical` expects logits, not probabilities.

Practical intuition:
- Logits are unnormalized scores.
- Softmax converts logits into probabilities.
- `categorical` already handles logits internally, so passing softmax output changes the intended sampling behavior.

Observed training quirk:
- When sampling used probabilities instead of logits, generated text looked much noisier.
- After using logits directly, samples at similar loss/step became more interpretable.

Takeaway:
- In LM generation, check each API contract carefully. "Looks mathematically close" can still produce different and worse behavior.

## 2026-05-01: Prompt Diversity Exposes Model Behavior Better

Using one fixed prompt can overestimate quality. A small prompt set gives a clearer view of generalization.

Practical intuition:
- One prompt can hide repetition loops or mode collapse.
- Multiple prompts test whether coherence holds across different starts.

Observed training quirk:
- Model looked strong on a single "Once upon a time" prompt.
- Multi-prompt sampling revealed where outputs still repeat phrases.

Takeaway:
- Always inspect multiple prompts at eval time before claiming generation quality.

## 2026-05-01: Repetition Penalty for Char-Level Decoding

Char-level models tend to repeat high-probability phrases during generation.

Practical intuition:
- A repetition penalty lowers logits for tokens already used in the current sequence.
- This reduces immediate loops while keeping base sampling logic unchanged.

Observed training quirk:
- Without penalty, coherent text still showed repeated templates.
- Adding penalty improved continuation variety while keeping readability.

Takeaway:
- For small char-level LMs, decoding controls can significantly improve perceived quality even when train/val loss changes little.

## 2026-05-01: Char-Level Baseline Is Stable Enough To Move On

Observed benchmark outcome:
- `latest` checkpoint reached val loss around `0.776` and val perplexity around `2.17`.
- Seed sweep variance is low (val loss std around `0.0021`, val ppl std around `0.0046`).

Practical implication:
- Remaining issues are mostly repetition/style limits of char-level tokenization rather than optimization instability.
- This is a good cutoff point to transition to BPE for higher quality/token efficiency.

## 2026-05-01: BPE Vocab Size Tradeoff Is Strong

Observed benchmark outcome at 8k steps:
- BPE-4096: val loss around `2.239`, val ppl around `9.387`, params `0.919M`.
- BPE-1024: val loss around `2.074`, val ppl around `7.953`, params `0.624M`.

Practical implication:
- Larger BPE vocab did not win under this parameter/training budget.
- BPE-1024 gave better quality-per-parameter and better runtime efficiency tradeoff.

## 2026-05-01: Longer BPE-1024 Training Still Pays Off

Observed benchmark outcome:
- BPE-1024 at 16k steps improved over 8k across val loss, val perplexity, bpc, and top-1 accuracy.

Practical implication:
- Under the same architecture and tokenizer, additional training budget still improves quality.
- It is worth finishing convergence checks on the chosen tokenizer regime before changing architecture again.
