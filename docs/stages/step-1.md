# Step 1: Loader Fix + Offline E2E Test

## What changed
- **models.py**: Added `NoSafetensorsError` exception. `_resolve_shard` now tries
  `model.safetensors.index.json` first, then `model.safetensors`, then raises a clear
  error instead of a confusing 404. Added `load_embeddings_from_dir` and
  `load_config_from_dir` for loading from local directories (used by tests).
- **cli.py**: `run_analysis` catches `NoSafetensorsError` and prints a user-friendly
  message instead of a traceback.
- **test_end_to_end.py**: 5 new tests building synthetic safetensors models in temp
  dirs with planted near-zero rows, verifying the full indicator pipeline.

## Why it exists
The original code assumed `model.safetensors` always exists. Repos like
`sshleifer/tiny-gpt2` only have `.bin` weights, causing a 404. The fix makes failure
explicit and recoverable, and the offline test proves the pipeline works without
any network access.

## What breaks if removed
- Repos without safetensors would crash with an opaque HTTP 404 instead of a clear error
- No way to test the full pipeline offline
- Batch runs couldn't gracefully skip unsupported models

## What tests prove and don't prove
**Prove**: planted near-zero embeddings rank in top 10 via L2 and cosine indicators;
sharded index resolution works; NoSafetensorsError is raised for missing files.
**Don't prove**: behavior with real HF Hub downloads; interaction with tokenizer_pass
(synthetic test doesn't use a real tokenizer).

## One thing that could still be wrong
The `load_embeddings_from_dir` function doesn't handle the case where sharded models
have different tensors in different shard files (it opens the same file for e_in and
e_out). For real sharded models this could fail if embeddings are in different shards.

---

### Quiz

1. Why does `_resolve_shard` try the index file before `model.safetensors`?
2. What happens if you call `load_embeddings_from_dir` on a directory with only
   a `pytorch_model.bin` file?
3. Why do the planted glitch tokens use norm ~1e-8 instead of exactly 0?

<details>
<summary>Answers</summary>

1. Sharded models don't have a single `model.safetensors`. The index file maps tensor
   names to specific shard files. Trying it first avoids downloading a file that doesn't
   exist.
2. `NoSafetensorsError` is raised with a message explaining the repo needs safetensors.
3. Exactly-zero rows are a degenerate case that some indicators handle specially (e.g.
   cosine distance assigns score 0.0). Using ~1e-8 tests the more realistic scenario
   where under-trained tokens have tiny but nonzero embeddings.
</details>
