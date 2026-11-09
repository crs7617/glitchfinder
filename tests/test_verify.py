"""Tests for verify.py — model-based verification."""

import pytest
import torch
from unittest.mock import MagicMock, patch

from glitchfinder.verify import (
    can_verify,
    _make_prompts,
    VerificationResult,
)
from glitchfinder.models import ModelInfo


class TestCanVerify:
    """Test model size gating."""

    def test_small_model_float32(self):
        info = ModelInfo(
            repo_id="test/small",
            vocab_size=32000,
            hidden_size=2048,
            tie_word_embeddings=True,
            architecture="LlamaForCausalLM",
            num_parameters=500_000_000,  # 500M
        )
        assert can_verify(info, "float32") is True

    def test_large_model_float32(self):
        info = ModelInfo(
            repo_id="test/large",
            vocab_size=32000,
            hidden_size=4096,
            tie_word_embeddings=True,
            architecture="LlamaForCausalLM",
            num_parameters=7_000_000_000,  # 7B
        )
        assert can_verify(info, "float32") is False

    def test_medium_model_bfloat16(self):
        """1.5B model should be verifiable in bfloat16 but not float32."""
        info = ModelInfo(
            repo_id="test/medium",
            vocab_size=32000,
            hidden_size=2048,
            tie_word_embeddings=True,
            architecture="LlamaForCausalLM",
            num_parameters=1_500_000_000,  # 1.5B
        )
        assert can_verify(info, "float32") is False
        assert can_verify(info, "bfloat16") is True

    def test_no_num_parameters_estimates(self):
        """When num_parameters is None, should estimate from vocab/hidden."""
        info = ModelInfo(
            repo_id="test/unknown",
            vocab_size=1000,
            hidden_size=64,
            tie_word_embeddings=True,
            architecture="GPT2LMHeadModel",
            num_parameters=None,
        )
        # 1000 * 64 * 4 = 256000, well under 1B
        assert can_verify(info, "float32") is True


class TestPromptTemplates:
    """Test prompt formatting."""

    def test_prompts_contain_token(self):
        prompts = _make_prompts("SolidGoldMagikarp")
        assert len(prompts) == 3
        for p in prompts:
            assert "SolidGoldMagikarp" in p

    def test_repetition_prompt_has_examples(self):
        prompts = _make_prompts("test")
        assert "Input:" in prompts[0]
        assert "Output:" in prompts[0]

    def test_qa_prompt_format(self):
        prompts = _make_prompts("test")
        assert "Q:" in prompts[1]
        assert "A:" in prompts[1]

    def test_infinite_repetition_long(self):
        prompts = _make_prompts("x")
        # 100 repetitions of "x "
        assert prompts[2].count("x") >= 100


class TestVerifyToken:
    """Test verification with mock model."""

    def test_probability_extraction(self):
        """Mock model returning fixed logits → verify probability math."""
        # This would require mocking torch model forward pass
        # Covered in integration tests with tiny-gpt2
        pass
