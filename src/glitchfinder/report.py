"""Report generation: build and save JSON results."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .models import ModelInfo
from .tokenizer_pass import TokenRecord
from .verify import VerificationResult


def _sanitize_filename(name: str) -> str:
    """Sanitize a model name for use as a filename.

    Replaces / with -- and removes other unsafe characters.
    """
    name = name.replace("/", "--")
    name = re.sub(r"[^\w\-.]", "_", name)
    return name


def compute_script_stats(
    records: list[TokenRecord],
    scores: Optional[dict[int, float]] = None,
    verification: Optional[dict[int, VerificationResult]] = None,
) -> dict:
    """Group tokens by Unicode script and count total/flagged/verified.

    Returns dict mapping script name to {total, flagged, verified_glitch}.
    """
    stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "flagged": 0, "verified_glitch": 0}
    )

    for r in records:
        scripts = r.unicode_scripts if r.unicode_scripts else {"Unknown"}
        is_flagged = r.is_special or r.is_unreachable or r.is_partial_utf8
        is_verified = (
            verification is not None
            and r.token_id in verification
            and verification[r.token_id].is_glitch
        )
        for script in scripts:
            stats[script]["total"] += 1
            if is_flagged:
                stats[script]["flagged"] += 1
            if is_verified:
                stats[script]["verified_glitch"] += 1

    return dict(stats)


def build_report(
    info: ModelInfo,
    records: list[TokenRecord],
    scores: dict[int, float],
    verification: Optional[dict[int, VerificationResult]],
    indicator_name: str,
    ranked_ids: Optional[list[int]] = None,
) -> dict:
    """Build the full JSON report structure.

    Args:
        info: Model metadata.
        records: Per-token analysis records.
        scores: Mapping of token_id to indicator score.
        verification: Mapping of token_id to verification result (if any).
        indicator_name: Name of the indicator used (e.g., "l2_norm").
        ranked_ids: Optional pre-ranked token IDs (most suspicious first).

    Returns:
        Dict ready for JSON serialization.
    """
    script_stats = compute_script_stats(records, scores, verification)

    # Build token list, ordered by rank if available
    token_order = ranked_ids if ranked_ids is not None else range(len(records))

    tokens = []
    for rank, tid in enumerate(token_order):
        tid = int(tid)
        if tid >= len(records):
            continue
        r = records[tid]

        token_entry = {
            "rank": rank,
            "token_id": r.token_id,
            "decoded": r.decoded_str,
            "raw_bytes_hex": r.raw_bytes.hex(),
            "score": float(scores.get(r.token_id, 0.0)),
            "is_special": r.is_special,
            "is_unreachable": r.is_unreachable,
            "is_partial_utf8": r.is_partial_utf8,
            "is_defined": r.is_defined,
            "unicode_scripts": sorted(r.unicode_scripts),
        }

        if verification and r.token_id in verification:
            v = verification[r.token_id]
            token_entry["verification"] = {
                "max_prob": v.max_prob,
                "is_glitch": v.is_glitch,
                "per_prompt_probs": v.per_prompt_probs,
            }

        tokens.append(token_entry)

    return {
        "metadata": {
            "repo_id": info.repo_id,
            "vocab_size": info.vocab_size,
            "hidden_size": info.hidden_size,
            "architecture": info.architecture,
            "tie_word_embeddings": info.tie_word_embeddings,
            "indicator": indicator_name,
        },
        "script_stats": script_stats,
        "tokens": tokens,
    }


def save_report(report: dict, output_dir: str, model_name: str) -> Path:
    """Save the report as JSON.

    Args:
        report: The report dict.
        output_dir: Directory to write to.
        model_name: Model identifier (will be sanitized for filename).

    Returns:
        Path to the written file.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    filename = _sanitize_filename(model_name) + ".json"
    file_path = out_path / filename

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return file_path
