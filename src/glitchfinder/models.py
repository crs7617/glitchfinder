"""Embedding extraction from safetensors without loading full models."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open

logger = logging.getLogger(__name__)


class NoSafetensorsError(Exception):
    """Raised when a repo has no safetensors weights."""
    pass

# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class ModelInfo:
    repo_id: str
    vocab_size: int
    hidden_size: int
    tie_word_embeddings: bool
    architecture: str
    num_parameters: Optional[int] = None


@dataclass
class Embeddings:
    info: ModelInfo
    e_in: np.ndarray
    e_out: Optional[np.ndarray]  # None when tied


# ── Tensor name mapping ─────────────────────────────────────────────────────

# Maps architecture name → {"e_in": tensor_name, "e_out": tensor_name | None}
_ARCH_TENSOR_MAP: dict[str, dict[str, Optional[str]]] = {
    "LlamaForCausalLM":   {"e_in": "model.embed_tokens.weight", "e_out": "lm_head.weight"},
    "MistralForCausalLM": {"e_in": "model.embed_tokens.weight", "e_out": "lm_head.weight"},
    "Qwen2ForCausalLM":   {"e_in": "model.embed_tokens.weight", "e_out": "lm_head.weight"},
    "GemmaForCausalLM":   {"e_in": "model.embed_tokens.weight", "e_out": "lm_head.weight"},
    "Gemma2ForCausalLM":  {"e_in": "model.embed_tokens.weight", "e_out": "lm_head.weight"},
    "PhiForCausalLM":     {"e_in": "model.embed_tokens.weight", "e_out": "lm_head.weight"},
    "Phi3ForCausalLM":    {"e_in": "model.embed_tokens.weight", "e_out": "lm_head.weight"},
    "Qwen3ForCausalLM":   {"e_in": "model.embed_tokens.weight", "e_out": "lm_head.weight"},
    "GPT2LMHeadModel":    {"e_in": "transformer.wte.weight", "e_out": None},
    "GPTNeoXForCausalLM": {"e_in": "gpt_neox.embed_in.weight", "e_out": "embed_out.weight"},
}


def _tensor_name_map(architecture: str, tied: bool) -> dict[str, Optional[str]]:
    """Return tensor names for the given architecture.

    If embeddings are tied, e_out is forced to None regardless of the
    architecture's default mapping.
    """
    if architecture in _ARCH_TENSOR_MAP:
        names = dict(_ARCH_TENSOR_MAP[architecture])
    else:
        # Fallback: common patterns for transformer models
        names = {"e_in": "model.embed_tokens.weight", "e_out": "lm_head.weight"}
        logger.warning(
            "Unknown architecture %s; guessing tensor names %s", architecture, names
        )
    if tied:
        names["e_out"] = None
    return names


def _scan_for_embed_tensor(keys: list[str], role: str) -> Optional[str]:
    """Fallback: scan safetensors keys for likely embedding tensors."""
    patterns = {
        "e_in": ["embed_tokens", "wte", "embed_in", "word_embeddings"],
        "e_out": ["lm_head", "embed_out", "output_embedding"],
    }
    for pattern in patterns.get(role, []):
        for key in keys:
            if pattern in key:
                return key
    return None


# ── Config loading ───────────────────────────────────────────────────────────

def load_config(repo_id: str, cache_dir: Optional[str] = None) -> ModelInfo:
    """Download and parse config.json from a HuggingFace repo."""
    config_path = hf_hub_download(
        repo_id, "config.json", cache_dir=cache_dir
    )
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)

    return _parse_config(repo_id, cfg)


def _parse_config(repo_id: str, cfg: dict) -> ModelInfo:
    """Parse a config dict into ModelInfo."""
    arch = cfg.get("architectures", ["Unknown"])[0]
    vocab_size = cfg["vocab_size"]
    hidden_size = cfg.get("hidden_size", cfg.get("n_embd", cfg.get("d_model", 0)))
    tied = cfg.get("tie_word_embeddings", True)  # default True per HF convention
    num_params = cfg.get("num_parameters")
    return ModelInfo(
        repo_id=repo_id,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        tie_word_embeddings=tied,
        architecture=arch,
        num_parameters=num_params,
    )


# ── Shard resolution ────────────────────────────────────────────────────────

def _resolve_shard(
    repo_id: str, tensor_name: str, cache_dir: Optional[str] = None
) -> Path:
    """Find which safetensors shard contains the given tensor.

    Resolution order:
    1. model.safetensors.index.json (sharded models) -- look up tensor in weight_map
    2. model.safetensors (single-file models)
    3. Raise NoSafetensorsError if neither exists (e.g. repo only has .bin weights)
    """
    # Try sharded index first
    try:
        index_path = hf_hub_download(
            repo_id, "model.safetensors.index.json", cache_dir=cache_dir
        )
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
        shard_file = index["weight_map"][tensor_name]
        shard_path = hf_hub_download(repo_id, shard_file, cache_dir=cache_dir)
        return Path(shard_path)
    except Exception:
        pass

    # Try single safetensors file
    try:
        path = hf_hub_download(repo_id, "model.safetensors", cache_dir=cache_dir)
        return Path(path)
    except Exception:
        pass

    raise NoSafetensorsError(
        f"No safetensors weights found for {repo_id}. "
        f"This repo may only have .bin weights, which are not supported. "
        f"Glitchfinder requires safetensors format."
    )


# ── Tensor loading ───────────────────────────────────────────────────────────

def load_embedding_tensor(
    repo_id: str, tensor_name: str, cache_dir: Optional[str] = None
) -> np.ndarray:
    """Load a single named tensor from safetensors as float32 numpy array.

    Uses torch framework for safe_open to handle bfloat16 tensors
    (numpy doesn't support bf16), then converts to float32 numpy.

    If the exact tensor_name isn't found, tries stripping the model prefix
    (e.g. "transformer.wte.weight" -> "wte.weight") since some safetensors
    files omit the HF model class prefix.
    """
    import torch
    shard_path = _resolve_shard(repo_id, tensor_name, cache_dir=cache_dir)
    with safe_open(str(shard_path), framework="pt") as f:
        keys = list(f.keys())
        # Try exact name first
        if tensor_name in keys:
            return f.get_tensor(tensor_name).float().numpy()
        # Try stripping prefix (e.g. "transformer.wte.weight" -> "wte.weight")
        short_name = tensor_name.rsplit(".", 2)[-2] + "." + tensor_name.rsplit(".", 1)[-1] if "." in tensor_name else tensor_name
        if short_name in keys:
            logger.info("Tensor %s not found, using %s", tensor_name, short_name)
            return f.get_tensor(short_name).float().numpy()
        # Try scanning for any key containing the leaf name
        leaf = tensor_name.split(".")[-2] + "." + tensor_name.split(".")[-1] if "." in tensor_name else tensor_name
        for k in keys:
            if k.endswith(leaf):
                logger.info("Tensor %s not found, using %s", tensor_name, k)
                return f.get_tensor(k).float().numpy()
        raise KeyError(
            f"Tensor {tensor_name} not found in {shard_path}. "
            f"Available keys: {keys[:20]}"
        )


def load_embeddings(
    repo_id: str, cache_dir: Optional[str] = None
) -> Embeddings:
    """Load input (and optionally output) embedding matrices.

    Only reads the embedding tensors via safetensors — never loads the
    full model. This keeps memory usage minimal.
    """
    info = load_config(repo_id, cache_dir=cache_dir)
    names = _tensor_name_map(info.architecture, info.tie_word_embeddings)

    # If we don't know the tensor name, try to scan the safetensors keys
    e_in_name = names["e_in"]
    e_out_name = names["e_out"]

    # Load input embeddings
    e_in = load_embedding_tensor(repo_id, e_in_name, cache_dir=cache_dir)

    # Validate vocab size vs tensor rows
    if e_in.shape[0] != info.vocab_size:
        actual = e_in.shape[0]
        logger.warning(
            "Vocab size mismatch: config says %d, tensor has %d rows. Using min.",
            info.vocab_size, actual,
        )
        use_size = min(info.vocab_size, actual)
        e_in = e_in[:use_size]
        info.vocab_size = use_size

    # Load output embeddings if untied
    e_out = None
    if e_out_name is not None:
        e_out = load_embedding_tensor(repo_id, e_out_name, cache_dir=cache_dir)
        if e_out.shape[0] != info.vocab_size:
            e_out = e_out[:info.vocab_size]

    return Embeddings(info=info, e_in=e_in, e_out=e_out)


# ── Local directory loading (for offline / test use) ─────────────────────

def load_config_from_dir(model_dir: str | Path) -> ModelInfo:
    """Load config.json from a local directory."""
    config_path = Path(model_dir) / "config.json"
    with open(config_path, encoding="utf-8") as f:
        cfg = json.load(f)
    return _parse_config(str(model_dir), cfg)


def load_embeddings_from_dir(model_dir: str | Path) -> Embeddings:
    """Load embeddings from a local directory containing safetensors + config.json.

    Same logic as load_embeddings but reads from disk instead of HuggingFace Hub.
    """
    model_dir = Path(model_dir)
    info = load_config_from_dir(model_dir)
    names = _tensor_name_map(info.architecture, info.tie_word_embeddings)

    e_in_name = names["e_in"]
    e_out_name = names["e_out"]

    # Find the safetensors file
    st_path = model_dir / "model.safetensors"
    if not st_path.exists():
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path, encoding="utf-8") as f:
                index = json.load(f)
            shard_file = index["weight_map"][e_in_name]
            st_path = model_dir / shard_file
        else:
            raise NoSafetensorsError(f"No safetensors in {model_dir}")

    import torch
    with safe_open(str(st_path), framework="pt") as f:
        e_in = f.get_tensor(e_in_name).float().numpy()

    if e_in.shape[0] != info.vocab_size:
        use_size = min(info.vocab_size, e_in.shape[0])
        e_in = e_in[:use_size]
        info.vocab_size = use_size

    e_out = None
    if e_out_name is not None:
        with safe_open(str(st_path), framework="pt") as f:
            e_out = f.get_tensor(e_out_name).float().numpy()
        if e_out.shape[0] != info.vocab_size:
            e_out = e_out[:info.vocab_size]

    return Embeddings(info=info, e_in=e_in, e_out=e_out)
