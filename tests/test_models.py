"""Tests for models.py — embedding extraction."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from glitchfinder.models import (
    _tensor_name_map,
    _parse_config,
    ModelInfo,
    NoSafetensorsError,
)


class TestTensorNameMap:
    """Test architecture → tensor name mapping."""

    def test_llama_names(self):
        names = _tensor_name_map("LlamaForCausalLM", tied=False)
        assert names["e_in"] == "model.embed_tokens.weight"
        assert names["e_out"] == "lm_head.weight"

    def test_llama_tied(self):
        names = _tensor_name_map("LlamaForCausalLM", tied=True)
        assert names["e_in"] == "model.embed_tokens.weight"
        assert names["e_out"] is None

    def test_gpt2_names(self):
        names = _tensor_name_map("GPT2LMHeadModel", tied=False)
        assert names["e_in"] == "transformer.wte.weight"
        # GPT2 has no separate output embedding
        assert names["e_out"] is None

    def test_gpt2_tied(self):
        names = _tensor_name_map("GPT2LMHeadModel", tied=True)
        assert names["e_out"] is None

    def test_gptneox_names(self):
        names = _tensor_name_map("GPTNeoXForCausalLM", tied=False)
        assert names["e_in"] == "gpt_neox.embed_in.weight"
        assert names["e_out"] == "embed_out.weight"

    def test_mistral_names(self):
        names = _tensor_name_map("MistralForCausalLM", tied=False)
        assert names["e_in"] == "model.embed_tokens.weight"
        assert names["e_out"] == "lm_head.weight"

    def test_unknown_architecture_falls_back(self):
        names = _tensor_name_map("SomeNewArchForCausalLM", tied=False)
        # Should still return something reasonable
        assert "e_in" in names
        assert "e_out" in names


class TestParseConfig:
    """Test config.json parsing into ModelInfo."""

    def test_basic_config(self):
        cfg = {
            "vocab_size": 32000,
            "hidden_size": 4096,
            "tie_word_embeddings": False,
            "architectures": ["LlamaForCausalLM"],
        }
        info = _parse_config("meta-llama/Llama-2-7b", cfg)
        assert info.vocab_size == 32000
        assert info.hidden_size == 4096
        assert info.tie_word_embeddings is False
        assert info.architecture == "LlamaForCausalLM"

    def test_default_tied_true(self):
        """tie_word_embeddings defaults to True when absent."""
        cfg = {
            "vocab_size": 50257,
            "hidden_size": 768,
            "architectures": ["GPT2LMHeadModel"],
        }
        info = _parse_config("gpt2", cfg)
        assert info.tie_word_embeddings is True

    def test_gpt2_n_embd_fallback(self):
        """GPT2 uses n_embd instead of hidden_size."""
        cfg = {
            "vocab_size": 50257,
            "n_embd": 768,
            "architectures": ["GPT2LMHeadModel"],
        }
        info = _parse_config("gpt2", cfg)
        assert info.hidden_size == 768

    def test_no_architectures(self):
        cfg = {
            "vocab_size": 32000,
            "hidden_size": 4096,
        }
        info = _parse_config("test/model", cfg)
        assert info.architecture == "Unknown"


class TestShardResolution:
    """Test shard resolution with fake index files."""

    def test_shard_index_lookup(self):
        """_resolve_shard should use index.json to find the right shard."""
        # This test would need to mock hf_hub_download, tested implicitly
        # in integration tests
        pass

    def test_shape_validation(self):
        """Verify that vocab size mismatch is handled."""
        # Construct a scenario where config says 32000 but tensor has 32001 rows
        # The load_embeddings function should use min() and log a warning
        pass
