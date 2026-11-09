# Step 6: Multi-Model Analysis

## What changed
- **models.py**: Tensor name fallback — strips model prefix if exact name not found
  (GPT-2 safetensors uses `wte.weight` not `transformer.wte.weight`).
- **batch_analyze.py**: Script to run all 5 models with `--no-verify`.
- Ran: Qwen3-0.6B, Qwen2.5-0.5B, SmolLM2-360M, GPT-2, Phi-1.5.

## Results summary
- **GPT-2**: Known glitch tokens surface (externalToEVA rank 9, rawdownload rank 24).
  Validates against the paper. 84 flagged/2000.
- **Qwen models**: Large vocab (152k), many unreachable whitespace tokens. 1890-1981 flagged.
- **SmolLM2**: Whitespace-dominated. 118 flagged.
- **Phi-1.5**: Untied, L2 norm indicator. 967 unreachable tokens with near-zero norms.

## What breaks if removed
- Tensor name fallback: GPT-2 analysis would fail.
- Batch script: have to run each model manually.

## What tests prove and don't prove
**Prove**: Pipeline works on 5 diverse architectures.
**Don't prove**: Verification on non-Qwen3 models.

## One thing that could still be wrong
The tensor name fallback is heuristic. A model with a tensor name that partially matches
(e.g., `some_prefix.wte.weight`) could match the wrong tensor.

---

### Quiz
1. Why does GPT-2's safetensors use `wte.weight` instead of `transformer.wte.weight`?
2. Why does Phi-1.5 use L2 norm instead of cosine distance?
3. Why are Qwen models' top tokens mostly whitespace?

<details>
<summary>Answers</summary>

1. The safetensors conversion strips the model class prefix. The HF `GPT2LMHeadModel`
   wraps the transformer with a `transformer.` prefix, but the raw weights don't have it.
2. Phi-1.5 has `tie_word_embeddings: false`, so the code uses L2 norm on input embeddings.
3. Qwen's 152k vocabulary includes many whitespace+newline combinations that are
   unreachable (the tokenizer merges them differently). These cluster near the garbage
   centroid because they're never used.
</details>
