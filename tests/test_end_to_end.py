"""Offline end-to-end test: build a tiny synthetic model, run the full pipeline,
assert planted under-trained tokens rank on top."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from glitchfinder.models import (
    load_embeddings_from_dir,
    load_config_from_dir,
    NoSafetensorsError,
)
from glitchfinder.indicators import (
    l2_norm_indicator,
    cosine_distance_indicator,
    rank_candidates,
)


VOCAB_SIZE = 100
HIDDEN_DIM = 32
# These token IDs will have near-zero embeddings (planted glitches)
PLANTED_GLITCH_IDS = [7, 42, 88]


def _build_synthetic_model(tmpdir: Path, tied: bool = True) -> Path:
    """Create a minimal model directory with config.json + model.safetensors.

    Normal tokens get random embeddings with norm ~5.
    Planted glitch tokens get near-zero embeddings (norm ~1e-8).
    """
    model_dir = tmpdir / "synthetic_model"
    model_dir.mkdir()

    rng = np.random.RandomState(12345)
    e_in = rng.randn(VOCAB_SIZE, HIDDEN_DIM).astype(np.float32) * 5.0

    # Plant near-zero rows for glitch tokens
    for gid in PLANTED_GLITCH_IDS:
        e_in[gid] = rng.randn(HIDDEN_DIM).astype(np.float32) * 1e-8

    tensors = {"transformer.wte.weight": e_in}
    save_file(tensors, str(model_dir / "model.safetensors"))

    config = {
        "architectures": ["GPT2LMHeadModel"],
        "vocab_size": VOCAB_SIZE,
        "n_embd": HIDDEN_DIM,
        "tie_word_embeddings": tied,
    }
    with open(model_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f)

    return model_dir


class TestOfflineEndToEnd:
    """Full pipeline on synthetic data -- no network, no HF downloads."""

    def test_planted_glitches_rank_top_with_l2(self):
        """For a tied GPT2-style model, L2 norm should surface planted zeros."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = _build_synthetic_model(Path(tmpdir), tied=True)
            emb = load_embeddings_from_dir(model_dir)

            assert emb.e_in.shape == (VOCAB_SIZE, HIDDEN_DIM)
            assert emb.e_out is None  # GPT2 + tied => no e_out
            assert emb.info.architecture == "GPT2LMHeadModel"

            # L2 norm indicator (tied model uses this)
            scores = l2_norm_indicator(emb.e_in)
            ranked = rank_candidates(scores, lower_is_glitchy=True)

            # All planted glitch IDs should appear in the top 10
            top_10 = set(ranked[:10].tolist())
            for gid in PLANTED_GLITCH_IDS:
                assert gid in top_10, (
                    f"Planted glitch token {gid} not in top 10: {sorted(top_10)}"
                )

    def test_planted_glitches_rank_top_with_cosine(self):
        """For an untied model, cosine distance from a planted reference cluster
        should surface tokens in that cluster."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = _build_synthetic_model(Path(tmpdir), tied=False)

            # For untied GPT2, e_out is None in the map, so we test cosine
            # directly on e_in using the planted zeros as the reference direction
            emb = load_embeddings_from_dir(model_dir)

            # Use the planted glitch IDs themselves as reference (they cluster
            # near zero). Other near-zero rows should also be close.
            scores = cosine_distance_indicator(emb.e_in, PLANTED_GLITCH_IDS)
            ranked = rank_candidates(scores, lower_is_glitchy=True)

            # The planted IDs should be in the top 10 (distance ~0 to their own centroid)
            top_10 = set(ranked[:10].tolist())
            for gid in PLANTED_GLITCH_IDS:
                assert gid in top_10, (
                    f"Planted glitch token {gid} not in top 10: {sorted(top_10)}"
                )

    def test_config_parsed_correctly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = _build_synthetic_model(Path(tmpdir))
            info = load_config_from_dir(model_dir)
            assert info.vocab_size == VOCAB_SIZE
            assert info.hidden_size == HIDDEN_DIM
            assert info.tie_word_embeddings is True
            assert info.architecture == "GPT2LMHeadModel"

    def test_no_safetensors_raises(self):
        """Directory without model.safetensors should raise NoSafetensorsError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "empty_model"
            model_dir.mkdir()
            config = {
                "architectures": ["GPT2LMHeadModel"],
                "vocab_size": 10,
                "n_embd": 8,
            }
            with open(model_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(config, f)

            with pytest.raises(NoSafetensorsError):
                load_embeddings_from_dir(model_dir)

    def test_sharded_index_resolution(self):
        """If model.safetensors.index.json exists, use it to find the shard."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "sharded_model"
            model_dir.mkdir()

            rng = np.random.RandomState(99)
            e_in = rng.randn(20, 8).astype(np.float32)
            tensors = {"transformer.wte.weight": e_in}
            shard_name = "model-00001-of-00001.safetensors"
            save_file(tensors, str(model_dir / shard_name))

            index = {
                "metadata": {},
                "weight_map": {"transformer.wte.weight": shard_name},
            }
            with open(model_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
                json.dump(index, f)

            config = {
                "architectures": ["GPT2LMHeadModel"],
                "vocab_size": 20,
                "n_embd": 8,
                "tie_word_embeddings": True,
            }
            with open(model_dir / "config.json", "w", encoding="utf-8") as f:
                json.dump(config, f)

            emb = load_embeddings_from_dir(model_dir)
            assert emb.e_in.shape == (20, 8)
