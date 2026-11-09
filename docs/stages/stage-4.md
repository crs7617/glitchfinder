# Stage 4: verify.py — Model-Based Verification

## Files

### `src/glitchfinder/verify.py`
**What it does:** The only module that loads the full model. Gates on size (<=1B float32, <=1.7B bfloat16), loads with `AutoModelForCausalLM`, then tests each candidate token against 3 prompt templates to see if the model ever predicts it with meaningful probability.

**Key functions:**
- `can_verify(info, dtype)` — Size gate. Checks num_parameters against threshold. If unknown, estimates from vocab_size * hidden_size * 4. Returns False for models too large to fit in RAM.
- `_make_prompts(token_str)` — Generates 3 prompts: (1) device repetition with input/output pairs, (2) Q&A "what does X mean?", (3) infinite repetition of the token 100 times. These are designed to give the model every chance to predict the token.
- `load_verification_model(repo_id, dtype)` — Downloads and loads the full model + tokenizer. Sets `.eval()` mode.
- `verify_token(model, tokenizer, token_id)` — For each prompt: encode, forward pass, softmax, extract p(token_id). Then generates 3 more steps greedily, checking probability at each. Returns `VerificationResult(max_prob, is_glitch=max_prob < 0.01)`.
- `verify_candidates(repo_id, candidate_ids, info, dtype)` — Orchestrates: gate → load → iterate → free model with `del + gc.collect()`.

**What breaks if removed:** No verification. All candidates would be "unverified" in the report. The pipeline still works (--no-verify flag), but you lose the confirmation step.

### `tests/test_verify.py`
9 tests covering:
- `can_verify`: small model (500M) OK, large model (7B) rejected, medium (1.5B) OK in bf16 but not f32, estimation from vocab*hidden when num_parameters is None
- Prompt templates: all 3 contain the token string, repetition prompt has I/O examples, Q&A has Q:/A:, infinite repetition has 100+ copies

**What tests prove:** Size gating logic is correct. Prompts are well-formed and contain the target token.
**What tests don't prove:** Actual model loading and forward pass (would need a real model download). The probability extraction math on real logits. Memory lifecycle (del model + gc.collect actually freeing RAM).

## One thing that could still be wrong
The verification uses greedy generation (pick most likely next token) to extend the context. This biases toward high-probability continuations. If a glitch token only becomes predictable after a specific non-greedy prefix, we'd miss it. Beam search or sampling would be more thorough but much slower.

---

## Questions

1. The 1% threshold (`max_prob < 0.01`) for glitch classification is from the paper. But some legitimate rare tokens (e.g., obscure Unicode characters) might also never exceed 1%. Should we add a secondary check — e.g., also require low L2 norm — or is the 1% threshold alone sufficient?

2. `verify_token` does a full forward pass for every generated step (no KV cache). For 3 prompts × (1 initial + 3 generation steps) = 12 forward passes per token, with 500 candidates that's 6000 forward passes. On CPU this could be very slow. Should we enable KV caching via `use_cache=True` and pass `past_key_values`?

3. The memory lifecycle deletes the model after verification but relies on `gc.collect()`. Python's garbage collector doesn't always release torch memory immediately. Should we add `torch.cuda.empty_cache()` (no-op on CPU but future-proof) or is `gc.collect()` sufficient for CPU-only?

<details><summary>Suggested answers</summary>

1. The 1% threshold alone is sufficient for classification. The paper validates this threshold empirically. Rare legitimate tokens would need to be filtered out at the candidate selection stage (they wouldn't have low indicator scores), not at verification. The two-stage approach (indicator → verification) already handles this.

2. Yes, KV caching would help significantly. Each generation step currently reprocesses the entire sequence. With `use_cache=True` and passing `past_key_values`, the incremental cost drops from O(seq_len) to O(1) per step. Worth adding as an optimization if verification proves too slow.

3. `gc.collect()` is sufficient for CPU. PyTorch CPU tensors are backed by regular malloc, which Python's GC releases. `torch.cuda.empty_cache()` is a CUDA-only concept. The real risk is if something else holds a reference to the model (e.g., a closure or exception traceback), preventing GC.

</details>
