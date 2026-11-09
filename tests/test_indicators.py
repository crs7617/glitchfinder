"""Tests for indicators.py — written BEFORE implementation (test-first).

These tests use synthetic embedding matrices with known properties
to verify that each indicator correctly identifies suspicious tokens.
"""

import numpy as np
import pytest

from glitchfinder.indicators import (
    l2_norm_indicator,
    cosine_distance_indicator,
    euclidean_distance_indicator,
    rank_candidates,
    select_verification_candidates,
)
from glitchfinder.tokenizer_pass import TokenRecord


def _make_record(tid, special=False, unreachable=False, partial=False):
    return TokenRecord(
        token_id=tid,
        decoded_str=f"token_{tid}",
        raw_bytes=b"",
        is_special=special,
        is_partial_utf8=partial,
        is_unreachable=unreachable,
    )


class TestL2NormIndicator:
    """L2 norm indicator: low norm → suspicious."""

    def test_near_zero_rows_ranked_lowest(self):
        """Rows at indices 5, 10, 15 are near-zero; they should rank in bottom 5."""
        rng = np.random.RandomState(42)
        e = rng.randn(100, 64).astype(np.float32)
        # Plant near-zero rows
        e[5] = 1e-8
        e[10] = 1e-9
        e[15] = 0.0

        norms = l2_norm_indicator(e)
        assert norms.shape == (100,)
        assert norms.dtype == np.float32

        # The near-zero rows should have the smallest norms
        ranked = np.argsort(norms)  # ascending
        bottom_5 = set(ranked[:5].tolist())
        assert {5, 10, 15}.issubset(bottom_5)

    def test_output_shape(self):
        e = np.ones((50, 32), dtype=np.float32)
        norms = l2_norm_indicator(e)
        assert norms.shape == (50,)

    def test_all_zero(self):
        e = np.zeros((10, 16), dtype=np.float32)
        norms = l2_norm_indicator(e)
        assert np.all(norms == 0.0)


class TestCosineDistanceIndicator:
    """Cosine distance from reference centroid: low distance → suspicious."""

    def test_planted_same_direction_rows(self):
        """Rows pointing in the same direction as reference tokens should
        have low cosine distance (= suspicious)."""
        rng = np.random.RandomState(42)
        e = rng.randn(100, 64).astype(np.float32)

        # Reference tokens: IDs 0-4, all point in the [1,1,1,...] direction
        ref_direction = np.ones(64, dtype=np.float32)
        ref_direction /= np.linalg.norm(ref_direction)
        for i in range(5):
            e[i] = ref_direction * (0.1 + rng.rand() * 0.1)

        # Plant 3 suspicious tokens in the same direction
        e[50] = ref_direction * 0.05
        e[60] = ref_direction * 0.08
        e[70] = ref_direction * 0.03

        ref_ids = [0, 1, 2, 3, 4]
        scores = cosine_distance_indicator(e, ref_ids)

        assert scores.shape == (100,)
        assert scores.dtype == np.float32

        # Planted tokens should have very low cosine distance
        # (close to 0, meaning they point in the same direction as refs)
        assert scores[50] < 0.1
        assert scores[60] < 0.1
        assert scores[70] < 0.1

    def test_zero_norm_row_no_nan(self):
        """A zero-norm row should produce score 0.0, not NaN."""
        e = np.ones((10, 8), dtype=np.float32)
        e[5] = 0.0  # zero row

        ref_ids = [0, 1, 2]
        scores = cosine_distance_indicator(e, ref_ids)

        assert not np.any(np.isnan(scores))
        assert scores[5] == 0.0  # maximally suspicious

    def test_single_reference_token(self):
        """Should work with just one reference token."""
        rng = np.random.RandomState(42)
        e = rng.randn(20, 8).astype(np.float32)
        ref_ids = [0]
        scores = cosine_distance_indicator(e, ref_ids)
        assert scores.shape == (20,)
        assert not np.any(np.isnan(scores))

    def test_orthogonal_rows_high_distance(self):
        """Rows orthogonal to reference should have distance ~1.0."""
        e = np.zeros((10, 4), dtype=np.float32)
        # Reference points in [1,0,0,0]
        e[0] = [1, 0, 0, 0]
        # Orthogonal row points in [0,1,0,0]
        e[5] = [0, 1, 0, 0]
        # Same direction
        e[6] = [2, 0, 0, 0]

        scores = cosine_distance_indicator(e, [0])
        assert abs(scores[5] - 1.0) < 0.01  # orthogonal → distance ~1
        assert scores[6] < 0.01              # same direction → distance ~0


class TestEuclideanDistanceIndicator:
    """Euclidean distance from reference centroid."""

    def test_basic_functionality(self):
        rng = np.random.RandomState(42)
        e = rng.randn(100, 64).astype(np.float32)

        ref_ids = [0, 1, 2, 3, 4]
        scores = euclidean_distance_indicator(e, ref_ids)
        assert scores.shape == (100,)
        assert scores.dtype == np.float32

    def test_close_rows_small_distance(self):
        """Rows near the reference centroid should have small distance."""
        e = np.zeros((20, 4), dtype=np.float32)
        # Reference tokens at [1,1,1,1]
        for i in range(5):
            e[i] = [1, 1, 1, 1]
        # Close token
        e[10] = [1.01, 1.01, 1.01, 1.01]
        # Far token
        e[15] = [100, 100, 100, 100]

        scores = euclidean_distance_indicator(e, [0, 1, 2, 3, 4])
        assert scores[10] < scores[15]


class TestRankCandidates:
    """Test ranking by suspiciousness."""

    def test_lower_is_glitchy(self):
        scores = np.array([5.0, 1.0, 3.0, 0.5, 4.0], dtype=np.float32)
        ranked = rank_candidates(scores, lower_is_glitchy=True)
        assert ranked[0] == 3  # score 0.5 is most suspicious
        assert ranked[1] == 1  # score 1.0

    def test_higher_is_glitchy(self):
        scores = np.array([5.0, 1.0, 3.0, 0.5, 4.0], dtype=np.float32)
        ranked = rank_candidates(scores, lower_is_glitchy=False)
        assert ranked[0] == 0  # score 5.0 is most suspicious

    def test_output_contains_all_ids(self):
        scores = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ranked = rank_candidates(scores, lower_is_glitchy=True)
        assert set(ranked.tolist()) == {0, 1, 2}


class TestSelectVerificationCandidates:
    """Test candidate selection with exclusion of flagged tokens."""

    def test_excludes_flagged(self):
        """Special, unreachable, and partial tokens should be excluded."""
        records = [
            _make_record(0, special=True),
            _make_record(1, unreachable=True),
            _make_record(2, partial=True),
            _make_record(3),
            _make_record(4),
            _make_record(5),
            _make_record(6),
            _make_record(7),
            _make_record(8),
            _make_record(9),
        ]
        # Rank: 0,1,2,3,4,5,6,7,8,9 (most suspicious first)
        ranked = np.arange(10, dtype=np.int64)
        candidates = select_verification_candidates(
            ranked, records, top_fraction=0.5
        )
        # 0,1,2 are flagged → excluded. 7 non-flagged tokens.
        # 50% of 7 = 3
        assert len(candidates) == 3
        assert 0 not in candidates
        assert 1 not in candidates
        assert 2 not in candidates

    def test_respects_rank_order(self):
        """Candidates should be in rank order (most suspicious first)."""
        records = [_make_record(i) for i in range(100)]
        ranked = np.arange(100, dtype=np.int64)
        candidates = select_verification_candidates(
            ranked, records, top_fraction=0.05
        )
        # Should be first 5% of 100 = 5 tokens
        assert candidates == [0, 1, 2, 3, 4]

    def test_top_fraction_default(self):
        records = [_make_record(i) for i in range(1000)]
        ranked = np.arange(1000, dtype=np.int64)
        candidates = select_verification_candidates(ranked, records)
        # Default 2% of 1000 = 20
        assert len(candidates) == 20
