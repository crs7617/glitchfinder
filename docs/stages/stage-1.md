# Stage 1: models.py — Embedding Extraction

## Files

### `src/glitchfinder/models.py`
**What it does:** Downloads and reads embedding matrices from HuggingFace repos using only safetensors (never loads the full model). This is the memory-critical module — it ensures we stay within 8GB RAM.

**Key functions:**
- `load_config(repo_id)` — downloads `config.json`, parses vocab_size, hidden_size, architecture, tie_word_embeddings into a `ModelInfo` dataclass. Every downstream module needs this metadata.
- `_tensor_name_map(architecture, tied)` — maps architecture strings (LlamaForCausalLM, GPT2LMHeadModel, etc.) to the actual tensor key names in the safetensors file. Without this, we wouldn't know which tensor to read.
- `_resolve_shard(repo_id, tensor_name)` — handles sharded models by reading `model.safetensors.index.json` to find which shard file contains a given tensor. Falls back to single-file models.
- `load_embedding_tensor(repo_id, tensor_name)` — reads one tensor from safetensors as float32 numpy. This is the primitive operation.
- `load_embeddings(repo_id)` — orchestrates everything: config → tensor names → shard resolution → load. Returns `Embeddings(info, e_in, e_out)`.

**What breaks if removed:** The entire pipeline. No other module can get embeddings or model metadata.

### `tests/test_models.py`
13 tests covering:
- Tensor name mapping for 5 architectures (Llama, GPT2, GPTNeoX, Mistral, unknown fallback)
- Tied vs untied behavior (e_out forced to None when tied)
- Config parsing: basic, default tie=True, GPT2's `n_embd` fallback, missing architectures

**What tests prove:** Name mapping and config parsing logic is correct for known architectures.
**What tests don't prove:** Actual HF downloads, safetensors reading, shard resolution with real files, vocab size mismatch handling. Those are integration-level concerns.

## One thing that could still be wrong
The `_tensor_name_map` fallback for unknown architectures guesses `model.embed_tokens.weight` / `lm_head.weight`. If a model uses a completely different naming scheme (e.g., RWKV, Mamba), this will fail silently — it'll try to load a nonexistent tensor and crash.

---

## Questions

1. The code always casts tensors to float32 via `.astype(np.float32)`. For a 128k-vocab model with hidden_size=4096, that's ~2GB for just e_in. Should we add a float16 indicator mode, or is the constraint "only analyze models whose embeddings fit in RAM" acceptable?

2. `_resolve_shard` tries the index.json first, then falls back to a single file. It doesn't handle the case where neither exists (e.g., a model that only has `.bin` pytorch files, no safetensors). Should we add a pytorch fallback, or document "safetensors-only" as a hard requirement?

3. The `_scan_for_embed_tensor` function exists but is never actually called — it was intended as a fallback when `_tensor_name_map` returns a wrong guess. Should it be wired in (try the mapped name, on KeyError scan the file's keys), or is the current explicit mapping + crash-on-unknown sufficient?

<details><summary>Suggested answers</summary>

1. The 8GB constraint means ~2GB for e_in is fine (leaves room for e_out + indicators + OS). Models with 128k+ vocab AND 4096+ hidden are rare enough that "embeddings must fit in RAM" is a reasonable documented limitation. Float16 indicators would lose precision for marginal savings.

2. Safetensors-only is the right call. The CLAUDE.md constraint says "read only the embedding tensor(s) via safetensors safe_open." Virtually all recent open-weight models publish safetensors. Document the requirement in README.

3. Wire it in. A KeyError from `safe_open` is a bad user experience. Better to try the mapped name, catch KeyError, then scan keys and retry. This costs one extra `safe_open` call in the failure case only.

</details>
