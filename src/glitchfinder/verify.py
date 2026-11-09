"""Model-based verification of glitch token candidates.

Loads the full model (only for small models <=1.7B) and checks whether
candidate tokens can be predicted by the model in controlled prompts.
A token that never appears with significant probability is confirmed
as a glitch token.
"""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from .models import ModelInfo

logger = logging.getLogger(__name__)

# ── Dataclass ────────────────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    token_id: int
    max_prob: float
    is_glitch: bool  # max_prob < 0.01
    per_prompt_probs: list[float] = field(default_factory=list)


# ── Size gates ───────────────────────────────────────────────────────────────

# Approximate parameter thresholds for CPU verification
_MAX_PARAMS_F32 = 1_000_000_000    # 1B for float32
_MAX_PARAMS_BF16 = 1_700_000_000   # 1.7B for bfloat16


def can_verify(info: ModelInfo, dtype: str = "float32") -> bool:
    """Check if the model is small enough to load for verification.

    Args:
        info: Model metadata.
        dtype: "float32" or "bfloat16".

    Returns:
        True if the model can be loaded on CPU with the given dtype.
    """
    # Estimate params from vocab_size * hidden_size if num_parameters unknown
    params = info.num_parameters
    if params is None:
        # Very rough estimate: most params are in attention/FFN layers,
        # but we don't know the exact architecture. Use vocab*hidden as
        # a lower bound and multiply by a factor.
        params = info.vocab_size * info.hidden_size * 4  # rough estimate

    if dtype == "bfloat16":
        return params <= _MAX_PARAMS_BF16
    return params <= _MAX_PARAMS_F32


# ── Prompt templates ─────────────────────────────────────────────────────────

def _make_prompts(token_str: str) -> list[str]:
    """Generate 3 verification prompts for a token string.

    These prompts are designed to elicit the token from the model if it
    has learned any meaningful association for it.

    1. Device repetition: input/output pairs that prime the model to repeat
    2. Q&A: asking what the token means
    3. Infinite repetition: many copies to see if the model can continue
    """
    # Prompt 1: Device repetition with examples
    prompt1 = (
        f'Input: "hello" Output: "hello"\n'
        f'Input: "world" Output: "world"\n'
        f'Input: "foo" Output: "foo"\n'
        f'Input: "bar" Output: "bar"\n'
        f'Input: "{token_str}" Output: "'
    )

    # Prompt 2: Q&A format
    prompt2 = (
        f'Q: What is "dog"?\n'
        f'A: "dog" is a common English word for a canine.\n\n'
        f'Q: What is "{token_str}"?\n'
        f'A: "'
    )

    # Prompt 3: Infinite repetition
    repeated = (token_str + " ") * 100
    prompt3 = repeated

    return [prompt1, prompt2, prompt3]


# ── Verification ─────────────────────────────────────────────────────────────

def load_verification_model(
    repo_id: str,
    dtype: str = "float32",
    cache_dir: Optional[str] = None,
) -> tuple:
    """Load a causal LM for verification.

    Returns (model, tokenizer) tuple. Model is set to eval mode.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(repo_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
    )
    model.eval()
    return model, tokenizer


def verify_token(
    model: torch.nn.Module,
    tokenizer,
    token_id: int,
    num_generate_steps: int = 3,
) -> VerificationResult:
    """Verify whether a single token can be predicted by the model.

    For each of 3 prompt templates:
    - Encode the prompt
    - Run forward pass
    - Extract softmax probability of the target token at the last position
    - Generate a few more tokens and check probability at each step

    A token is classified as a glitch if its maximum probability across
    all prompts and generation steps is below 1%.

    Args:
        model: The loaded causal LM.
        tokenizer: The tokenizer.
        token_id: The token ID to verify.
        num_generate_steps: Number of additional tokens to generate per prompt.

    Returns:
        VerificationResult with max probability and glitch classification.
    """
    token_str = tokenizer.decode([token_id], skip_special_tokens=False)
    prompts = _make_prompts(token_str)
    per_prompt_probs: list[float] = []
    all_probs: list[float] = []

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        prompt_max_prob = 0.0

        with torch.no_grad():
            # Initial forward pass
            outputs = model(input_ids)
            logits = outputs.logits[0, -1, :]  # last token's logits
            probs = torch.softmax(logits, dim=0)
            p = probs[token_id].item()
            prompt_max_prob = max(prompt_max_prob, p)

            # Generate a few more steps
            current_ids = input_ids
            for _ in range(num_generate_steps):
                # Greedy: pick the most likely next token
                next_token = logits.argmax().unsqueeze(0).unsqueeze(0)
                current_ids = torch.cat([current_ids, next_token], dim=1)
                outputs = model(current_ids)
                logits = outputs.logits[0, -1, :]
                probs = torch.softmax(logits, dim=0)
                p = probs[token_id].item()
                prompt_max_prob = max(prompt_max_prob, p)

        per_prompt_probs.append(prompt_max_prob)
        all_probs.append(prompt_max_prob)

    max_prob = max(all_probs) if all_probs else 0.0
    return VerificationResult(
        token_id=token_id,
        max_prob=max_prob,
        is_glitch=max_prob < 0.01,
        per_prompt_probs=per_prompt_probs,
    )


def verify_candidates(
    repo_id: str,
    candidate_ids: list[int],
    info: ModelInfo,
    dtype: str = "float32",
    cache_dir: Optional[str] = None,
) -> list[VerificationResult]:
    """Verify a list of candidate tokens.

    Checks model size, loads the model once, iterates through candidates,
    then frees the model.

    Args:
        repo_id: HuggingFace model repo ID.
        candidate_ids: Token IDs to verify.
        info: Model metadata.
        dtype: "float32" or "bfloat16".
        cache_dir: Optional HF cache directory.

    Returns:
        List of VerificationResult, or empty list if model too large.
    """
    if not can_verify(info, dtype):
        logger.warning(
            "Model too large for verification with dtype=%s. "
            "Candidates will be marked as unverified.",
            dtype,
        )
        return [
            VerificationResult(
                token_id=tid, max_prob=-1.0, is_glitch=False,
                per_prompt_probs=[],
            )
            for tid in candidate_ids
        ]

    print(f"  Loading model for verification (dtype={dtype})...")
    model, tokenizer = load_verification_model(repo_id, dtype, cache_dir)

    results = []
    total = len(candidate_ids)
    for i, tid in enumerate(candidate_ids):
        if (i + 1) % 50 == 0 or i == 0 or i == total - 1:
            print(f"  Verifying token {i + 1}/{total}...")
        result = verify_token(model, tokenizer, tid)
        results.append(result)

    # Free model memory
    del model
    del tokenizer
    gc.collect()

    return results
