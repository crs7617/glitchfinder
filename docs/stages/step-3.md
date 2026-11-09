# Step 3: Real Run on Qwen3-0.6B

## What changed
- **models.py**: Added `Qwen3ForCausalLM` to `_ARCH_TENSOR_MAP`. Changed
  `load_embedding_tensor` to use `framework="pt"` for safe_open then `.float().numpy()`,
  fixing bfloat16 support (numpy can't read bf16 directly).

## Results
- Model: Qwen/Qwen3-0.6B, 151936 vocab, 1024 hidden, tied embeddings
- 14 special tokens, 4065 unreachable, 0 partial UTF-8
- 708 reference tokens used for cosine distance indicator
- Top 30 are all whitespace+newline patterns flagged as unreachable
- Top non-flagged suspects: rare Arabic ligatures (U+FC19 etc.), isolated Hangul
  syllables, an Egyptian hieroglyph, `webElementX` (Selenium artifact), and
  FIM tokens (`<|fim_suffix|>`, `<|fim_middle|>`)
- Peak RAM ~1.5 GB, well under the 3 GB limit

## Honest assessment
The top-30 results are dominated by unreachable whitespace tokens, which is
expected but not very interesting. The truly interesting candidates (rank 338+)
are rare script characters and code artifacts. Without verification, we can only
say these have low cosine distance to the unreachable cluster, not that they're
confirmed glitch tokens.

## What breaks if removed
- bfloat16 models (most modern HF models) would crash on loading
- Qwen3 architecture would fall back to guessed tensor names

## What tests prove and don't prove
**Prove**: bf16 loading works (via existing synthetic tests using float32).
**Don't prove**: actual bf16 precision (synthetic tests use float32 safetensors).

## One thing that could still be wrong
The cosine distance indicator for tied models compares to a reference centroid built
from unreachable/special tokens. This means the indicator partially recapitulates the
tokenizer heuristics rather than being fully independent.

---

### Quiz

1. Why can't numpy read bfloat16 tensors directly?
2. Why are whitespace+newline tokens "unreachable" in Qwen3?
3. What would change if Qwen3 had `tie_word_embeddings: false`?

<details>
<summary>Answers</summary>

1. bfloat16 is not a standard IEEE 754 format. numpy only supports float16/32/64.
   PyTorch has native bf16 support via `torch.bfloat16`.
2. The tokenizer merges whitespace differently during encoding vs decoding. Encoding
   "  \n" might produce different token IDs than the single token that decodes to it.
3. The indicator would switch from cosine_distance to l2_norm, using the input
   embedding matrix only. The lm_head.weight would also be loaded as e_out.
</details>
