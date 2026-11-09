"""Statistical indicators for detecting under-trained (glitch) tokens.

All functions are pure: arrays in, scores out. No side effects, no I/O.
Heavily commented to explain the reasoning behind each indicator.

Reference: arXiv 2405.05417 ("Fishing for Magikarp")
"""

from __future__ import annotations

import numpy as np


def l2_norm_indicator(e_in: np.ndarray) -> np.ndarray:
    """Compute L2 norm of each embedding row.

    Under-trained tokens tend to have smaller norms because weight decay
    pushes unused embeddings toward zero during training. A low L2 norm
    is therefore suspicious — the token may not have received enough
    gradient updates to develop a meaningful representation.

    Args:
        e_in: Input embedding matrix, shape (vocab_size, hidden_dim).

    Returns:
        Array of shape (vocab_size,) with L2 norms. Lower = more suspicious.
    """
    return np.linalg.norm(e_in, axis=1).astype(np.float32)


def cosine_distance_indicator(
    e_out: np.ndarray, ref_token_ids: list[int]
) -> np.ndarray:
    """Cosine distance from the mean of reference (known-unused) token embeddings.

    For tied-embedding models, glitch tokens cluster near the "garbage"
    direction defined by known-unused tokens (special tokens, unreachable
    tokens). We compute the mean embedding of reference tokens, then
    measure how close each token is to this mean via cosine similarity.

    Lower distance (closer to reference centroid) = more suspicious.

    Args:
        e_out: Output embedding matrix, shape (vocab_size, hidden_dim).
        ref_token_ids: IDs of known-unused tokens to form the reference centroid.

    Returns:
        Array of shape (vocab_size,) with cosine distances in [0, 2].
        Tokens with zero-norm embeddings get score 0.0 (maximally suspicious).
    """
    # Compute reference centroid: mean of reference token embeddings
    ref_embeddings = e_out[ref_token_ids]  # (n_ref, hidden_dim)
    u_ref = ref_embeddings.mean(axis=0)    # (hidden_dim,)

    # Norms for all rows and the reference centroid
    row_norms = np.linalg.norm(e_out, axis=1)     # (vocab_size,)
    ref_norm = np.linalg.norm(u_ref)               # scalar

    # Dot products with reference centroid
    dots = e_out @ u_ref                            # (vocab_size,)

    # Cosine similarity, handling zero-norm rows
    # cos_sim = dot(row, u_ref) / (||row|| * ||u_ref||)
    denom = row_norms * ref_norm
    # Where denom is zero, set cos_sim to 1.0 so distance = 0.0 (suspicious)
    safe_denom = np.where(denom > 0, denom, 1.0)
    cos_sim = dots / safe_denom

    # For zero-norm rows: cosine distance = 0.0 (maximally suspicious)
    cos_dist = 1.0 - cos_sim
    cos_dist = np.where(denom > 0, cos_dist, 0.0)

    return cos_dist.astype(np.float32)


def euclidean_distance_indicator(
    e_out: np.ndarray, ref_token_ids: list[int]
) -> np.ndarray:
    """Euclidean distance from reference centroid, chunked for memory efficiency.

    Similar logic to cosine_distance_indicator but uses L2 distance.
    Smaller distance = closer to the garbage cluster = more suspicious.

    Chunked in blocks of 16k rows to avoid doubling memory from broadcasting
    the full (vocab_size, hidden_dim) difference matrix.

    Args:
        e_out: Output embedding matrix, shape (vocab_size, hidden_dim).
        ref_token_ids: IDs of known-unused tokens.

    Returns:
        Array of shape (vocab_size,) with Euclidean distances.
    """
    ref_embeddings = e_out[ref_token_ids]
    u_ref = ref_embeddings.mean(axis=0)  # (hidden_dim,)

    vocab_size = e_out.shape[0]
    distances = np.empty(vocab_size, dtype=np.float32)

    # Process in chunks to keep memory usage bounded
    chunk_size = 16384
    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        chunk = e_out[start:end]  # (chunk_size, hidden_dim)
        diff = chunk - u_ref      # broadcasts u_ref
        distances[start:end] = np.linalg.norm(diff, axis=1)

    return distances


def rank_candidates(scores: np.ndarray, lower_is_glitchy: bool) -> np.ndarray:
    """Rank token IDs by suspiciousness.

    Args:
        scores: Per-token scores from an indicator function.
        lower_is_glitchy: If True, lower scores are more suspicious (sorted ascending).
                          If False, higher scores are more suspicious (sorted descending).

    Returns:
        Array of token IDs sorted by suspiciousness (most suspicious first).
    """
    if lower_is_glitchy:
        return np.argsort(scores).astype(np.int64)
    else:
        return np.argsort(-scores).astype(np.int64)


def select_verification_candidates(
    ranked_ids: np.ndarray,
    records: list,
    top_fraction: float = 0.02,
) -> list[int]:
    """Select the top candidates for verification, excluding already-flagged tokens.

    We skip tokens that are already known to be problematic (special,
    unreachable, partial UTF-8) since those don't need verification.
    From the remaining, we take the top fraction as verification candidates.

    Args:
        ranked_ids: Token IDs sorted by suspiciousness (most suspicious first).
        records: List of TokenRecord objects.
        top_fraction: Fraction of non-flagged tokens to select (default 2%).

    Returns:
        List of token IDs to verify.
    """
    # Build set of already-flagged token IDs
    flagged = set()
    for r in records:
        if r.is_special or r.is_unreachable or r.is_partial_utf8:
            flagged.add(r.token_id)

    # Filter out flagged tokens, preserving rank order
    clean = [tid for tid in ranked_ids if int(tid) not in flagged]

    # Take top fraction
    n = max(1, int(len(clean) * top_fraction))
    return clean[:n]
