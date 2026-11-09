# Stage 3: indicators.py — Statistical Indicators

## Files

### `src/glitchfinder/indicators.py`
**What it does:** The mathematical core. Pure functions that take numpy arrays and return per-token suspiciousness scores. No I/O, no side effects. Heavily commented to explain the reasoning.

**Key functions:**
- `l2_norm_indicator(e_in)` — Computes L2 norm of each row. Under-trained tokens get pushed toward zero by weight decay, so low norm = suspicious. Returns shape (vocab_size,).
- `cosine_distance_indicator(e_out, ref_token_ids)` — For tied models. Computes centroid of known-unused reference tokens, then measures cosine distance from each row to that centroid. Low distance = row points in the "garbage direction" = suspicious. Handles zero-norm rows (score 0.0, not NaN).
- `euclidean_distance_indicator(e_out, ref_token_ids)` — Same idea but L2 distance. Chunked in 16k-row blocks to avoid allocating a full (vocab_size, hidden_dim) difference matrix.
- `rank_candidates(scores, lower_is_glitchy)` — Sorts token IDs by score. Ascending if lower=suspicious, descending if higher=suspicious. Returns np.ndarray of ranked IDs.
- `select_verification_candidates(ranked_ids, records, top_fraction)` — Takes the top fraction of ranked tokens, excluding those already flagged (special, unreachable, partial UTF-8). These are the ones worth verifying with the full model.

**What breaks if removed:** No suspiciousness scores → no ranking → no candidates → no verification → no report. This is the analytical heart of the tool.

### `tests/test_indicators.py`
15 tests (written before implementation per plan):
- L2 norm: planted near-zero rows at indices 5,10,15 rank in bottom 5; output shape; all-zero matrix
- Cosine distance: planted same-direction rows detected; zero-norm → no NaN, score 0.0; single reference token; orthogonal rows → distance ~1.0
- Euclidean distance: basic shape; close rows < far rows
- Ranking: lower_is_glitchy and higher_is_glitchy orderings; all IDs present
- Candidate selection: excludes flagged tokens; preserves rank order; default 2% fraction

**What tests prove:** The math is correct for synthetic cases. Zero-norm edge case is safe. Flagged-token exclusion works.
**What tests don't prove:** Behavior on real embedding distributions. Whether cosine vs euclidean actually finds real glitch tokens. Chunking correctness at exact boundary (16384th row).

## One thing that could still be wrong
The cosine distance indicator sets zero-norm rows to score 0.0 (maximally suspicious). But a zero-norm row could also mean padding — e.g., if the model has `vocab_size=32000` but the tensor is padded to 32768 rows. Those padding rows aren't real glitch tokens. The `select_verification_candidates` exclusion would need to know about padding, which it currently doesn't.

---

## Questions

1. The paper (arXiv 2405.05417) uses cosine distance from the centroid of known-unused tokens as the primary indicator. Our implementation also offers L2 norm as an alternative for untied models. Is using L2 norm for untied models the right choice, or should we use cosine distance from reference tokens on the output embedding matrix (e_out) for untied models too?

2. The chunked euclidean distance uses a 16k-row block size. On an 8GB machine with float32, this means each chunk allocates ~16384 * hidden_dim * 4 bytes. For hidden_dim=4096, that's ~256MB per chunk. Is this acceptable, or should we shrink to 8k?

3. `select_verification_candidates` uses a flat 2% cutoff. For a 32k vocab, that's 640 tokens. For a 256k vocab (Llama 3), that's 5120 tokens — verification at ~3 prompts each would be very slow. Should we cap the absolute count (e.g., max 500), or is the CLI's `--top-n 2000` flag sufficient?

<details><summary>Suggested answers</summary>

1. For untied models, L2 norm on e_in is the simpler and cheaper indicator. The paper focuses on tied models where cosine distance is necessary because the output embedding space is shared. For untied models, L2 norm captures the weight-decay signal directly. Both are valid; L2 norm is the better default for untied.

2. 256MB per chunk is fine — we have 8GB total, embeddings are ~2GB max, leaving plenty. 16k is a good default. The chunking is about avoiding a full (vocab_size, hidden_dim) broadcast, not about keeping chunks small.

3. The CLI `--top-n 2000` is sufficient. The 2% is the initial selection, and `--top-n` caps what actually gets verified. Document that large vocab models should use a lower `--top-n` in README.

</details>
