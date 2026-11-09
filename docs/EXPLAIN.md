# How Glitchfinder Works

## What Are Glitch Tokens?

Large language models have a fixed vocabulary — a list of tokens (words, subwords, or byte sequences) that the model can read and produce. Most of these tokens get used heavily during training: the model sees them millions of times and learns meaningful representations for them.

But some tokens barely get used at all. Maybe they were added for coverage of rare Unicode blocks, or they existed in the tokenizer's training data but almost never appeared in the model's training data. These **under-trained tokens** develop degenerate embeddings — their vector representations carry little semantic information.

When you force the model to encounter these tokens, it often produces bizarre, incoherent, or repetitive output. The tokens are called **glitch tokens** (or "Magikarp" tokens, after the paper that formalized their study: arXiv 2405.05417, "Fishing for Magikarp").

## Tied vs Untied Embeddings

Transformer language models have two embedding matrices:

- **Input embeddings (e_in):** Maps token IDs to vectors that enter the transformer layers. Shape: (vocab_size, hidden_dim).
- **Output embeddings (e_out):** Maps the transformer's output back to token probabilities. Shape: (vocab_size, hidden_dim).

In **tied** models, these two matrices are literally the same object — `e_in == e_out`. This is common (GPT-2, most Llama variants with `tie_word_embeddings=True`). It saves parameters but means the embedding must serve both roles.

In **untied** models, they're separate. The input embedding can specialize for reading tokens; the output embedding can specialize for predicting them.

This distinction matters for glitch detection because:
- **Tied models:** A glitch token's single embedding is pulled in two competing directions (input and output) but trained on neither. It tends to cluster near other unused tokens in a "garbage direction." We detect this with **cosine distance from a reference centroid**.
- **Untied models:** The input embedding of a glitch token simply decays toward zero under weight decay. We detect this with **L2 norm**.

## Indicators

### L2 Norm (untied models)

The simplest indicator. During training, weight decay adds a penalty proportional to the squared magnitude of each parameter. For tokens that receive few or no gradient updates, weight decay pushes their embeddings toward zero. So: **low L2 norm = likely under-trained**.

```
score[i] = ||e_in[i]||_2
```

Lower scores are more suspicious.

### Cosine Distance from Reference Centroid (tied models)

For tied models, we first identify **reference tokens** — tokens we already know are unused:
- Special tokens (BOS, EOS, PAD) that are unreachable via normal tokenization
- Tokens with empty decoded forms
- Tokens that don't roundtrip: `encode(decode(id)) != [id]`

We compute the mean embedding of these reference tokens — the **reference centroid**. This represents the "garbage direction" in the embedding space.

Then for each token, we compute:
```
score[i] = 1 - cos_sim(e_out[i], u_ref)
```

where `cos_sim` is cosine similarity and `u_ref` is the reference centroid. Tokens close to the centroid (low cosine distance) are suspicious — they look like the known-unused tokens.

Zero-norm rows get a score of 0.0 (maximally suspicious) rather than NaN.

### Euclidean Distance from Reference Centroid

Same idea as cosine distance, but using L2 distance instead:
```
score[i] = ||e_out[i] - u_ref||_2
```

This captures both direction (like cosine) and magnitude. Computed in chunks of 16k rows to avoid doubling memory usage from broadcasting.

## Verification

Indicators produce candidates. Verification confirms them.

For models small enough to fit in RAM (<=1B parameters at float32), we load the full model and test each candidate with three prompt templates:

1. **Device repetition:** A series of "Input: X Output: X" pairs, ending with the candidate token. If the model can't predict the token even when primed to repeat it, that's a strong signal.

2. **Q&A:** "What does 'TOKEN' mean?" with an example answer. Tests whether the model has any semantic association.

3. **Infinite repetition:** The token repeated 100 times. Tests whether the model can continue a pattern of the token.

For each prompt, we run a forward pass, extract the softmax probability of the candidate token, then generate 3 more steps greedily. The **maximum probability across all prompts and steps** determines the classification:

- **max_prob < 1%** → Confirmed glitch token
- **max_prob >= 1%** → Not a glitch (the model can predict it in some context)

## Token Classification Flags

Before looking at embeddings, we classify tokens by structural properties:

- **Special:** Token is in the tokenizer's special token list (BOS, EOS, PAD, etc.)
- **Unreachable:** `tokenizer.encode(tokenizer.decode([id])) != [id]` — the token can't be reached by encoding any text
- **Partial UTF-8:** The token's raw bytes don't form valid UTF-8 (common in byte-fallback tokenizers like Llama)
- **Undefined:** Contains Unicode codepoints in the "Cn" (unassigned) category

These flags are used to:
1. Select reference tokens for the cosine distance indicator
2. Exclude already-known-problematic tokens from verification (saves time)
3. Enrich the final report with structural context

## Interpreting Results

The JSON report ranks tokens by suspiciousness. When reading results:

- **Rank 0-50:** Very likely glitch tokens if verified, or strong candidates if unverified
- **Verified glitch (max_prob < 1%):** High confidence. The model genuinely cannot predict this token.
- **Candidates, unverified:** The indicator says suspicious, but no model confirmation. Could be false positives (legitimate rare tokens with unusual embeddings).
- **Flagged tokens (special/unreachable/partial):** Already known to be problematic by structure alone. Not surprising and not very interesting.
- **Per-script stats:** Show which Unicode scripts have the most suspicious tokens. CJK, Cyrillic, and rare scripts often have more glitch tokens because they're underrepresented in English-centric training data.

## Limitations

- **Safetensors only.** Models that only publish PyTorch `.bin` files aren't supported.
- **CPU only, 8GB RAM.** Very large embedding matrices (128k+ vocab with 8192+ hidden dim) may not fit.
- **Verification limited to <=1B models.** Larger models are analyzed by indicators only.
- **Indicator heuristics, not guarantees.** L2 norm and cosine distance are strong signals but not perfect. A token with a low norm might be legitimate but rare. A token close to the garbage centroid might be there by coincidence.
- **Three prompt templates.** A glitch token that happens to be predictable in a context we don't test would be missed. The 1% threshold is empirically reasonable but arbitrary.

## Paper Correspondence

This section quotes arXiv 2405.05417 (Land & Bartolo, 2024) and maps each claim
to our implementation.

### Indicator selection (Section 2.2)

> "Thus, for models with tied embeddings, we use the cosine distance-based indicator
> C(E_out, u_ref) to select candidate tokens. For models without tied embeddings,
> we use the norm of E_in, denoted L2(E_in)."

Our code (`cli.py` lines 79-91): tied models use `cosine_distance_indicator(emb.e_in, ref_ids)`.
Since `e_in == e_out` for tied models, this is equivalent to the paper's `C(E_out, u_ref)`.
Untied models use `l2_norm_indicator(emb.e_in)`, matching `L2(E_in)`.

### Reference token selection (Section 2.2)

> "we start by defining a set of known untrained or highly under-trained embedding
> indices t_ref, e.g. the token IDs for tokens such as <unused_token123>, or the
> space of embeddings above the tokenizer vocabulary size."

Our code (`tokenizer_pass.py:find_reference_token_ids`): uses special+unreachable tokens,
empty-decode tokens, and partial UTF-8 as fallback. This is a reasonable approximation
of "known untrained" tokens, though the paper also mentions above-vocab-size indices
(which we don't use, since we truncate to vocab_size).

### Verification threshold (Section 2.3)

> "we verify all tokens that rank among the most likely 2% according to the chosen
> indicator, excluding partial UTF-8 sequences and unreachable tokens. This verification
> process involves constructing specific repetitive prompts that induce a high output
> probability for normal tokens, and checking if a particular candidate token has a
> very low (< 1%) output probability."

Our code: `select_verification_candidates` uses `top_fraction=0.02` (2%) and excludes
special, unreachable, and partial UTF-8 tokens. `verify_token` classifies as glitch if
`max_prob < 0.01` (1%). **Exact match.**

### Weight decay rationale (Section 2.2)

> "When embeddings are not tied, input embeddings for tokens which do not appear in
> the input for a training step are only affected by a potential weight decay term.
> If weight decay is applied to the input embedding matrix, the embeddings
> corresponding to under-trained tokens will tend to zero as training progresses."

This is why L2 norm works for untied models: weight decay pushes unused embeddings
toward zero.

> "untrained tokens will experience similar updates in training, 'moving away' from
> the mean output vector of the model. Thus, we can expect to find that under-trained
> token embeddings share a similar direction in output embedding space"

This is why cosine distance from a reference centroid works for tied models: under-trained
tokens cluster in a common "garbage direction."

## References

- Land, S., & Bartolo, M. (2024). "Fishing for Magikarp: Automatically Detecting Under-trained Tokens in Large Language Models." arXiv:2405.05417.
