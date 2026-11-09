# Step 4: Paper Check

## What changed
- **docs/EXPLAIN.md**: Added "Paper Correspondence" section with verbatim quotes from
  arXiv 2405.05417 (Land & Bartolo, 2024), mapping each paper claim to our code.

## Key findings
- **Indicator selection**: Paper says tied -> `C(E_out, u_ref)`, untied -> `L2(E_in)`.
  Our code matches exactly (for tied models, e_in == e_out so using e_in is equivalent).
- **Verification**: Paper says top 2%, exclude partial UTF-8 and unreachable, threshold
  < 1%. Our code: `top_fraction=0.02`, excludes flagged tokens, `max_prob < 0.01`. Match.
- **Reference tokens**: Paper mentions `<unused_token123>` and above-vocab indices. We
  use special+unreachable+empty-decode tokens. Reasonable approximation.

## What breaks if removed
Nothing functional — this is documentation only. But the project's credibility depends
on demonstrating that the implementation matches the published methodology.

## What tests prove and don't prove
**Prove**: N/A (no code changes).
**Don't prove**: Whether our reference token selection is optimal vs. the paper's.

## One thing that could still be wrong
For untied models, the paper also calculates output-embedding-based indicators as
supplementary data. We only use L2(E_in). Adding output indicators could catch cases
where weight decay wasn't applied to input embeddings (e.g., GPT-NeoX, OLMo v1).

---

### Quiz

1. Why does the paper use E_out for tied models but E_in for untied models?
2. What is the "garbage direction" the paper describes?
3. The paper notes GPT-NeoX has non-zero embeddings for untrained tokens. Why?

<details>
<summary>Answers</summary>

1. For tied models, the shared embedding is pulled in a common direction by output
   gradient updates — detectable by cosine distance. For untied models, unused input
   embeddings simply decay toward zero under weight decay — detectable by L2 norm.
2. The common direction in embedding space where under-trained tokens cluster. It
   emerges because all untrained tokens receive similar gradient updates (from the
   output/loss side) that push them away from the mean output vector.
3. GPT-NeoX uses a weight decay implementation where only weights used in the forward
   pass are affected. Untrained input embeddings are never in the forward pass, so
   weight decay doesn't touch them — they stay at initialization values.
</details>
