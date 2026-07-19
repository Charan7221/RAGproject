"""Tests for PostgreSQL hybrid retriever logic."""

import pytest
from src.rag_qa.core.pg_retriever import PgHybridRetriever


class TestReciprocalRankFusion:
    """Unit tests for RRF fusion without database dependencies."""

    def _make_retriever(self, rrf_k=60):
        return PgHybridRetriever(
            embedding_function=None,
            session_id="test-session",
            rrf_k=rrf_k
        )

    def test_rrf_boosts_docs_in_both_lists(self):
        retriever = self._make_retriever()
        semantic = [
            (1, 0.9, "content A", {}, "a.pdf"),
            (2, 0.8, "content B", {}, "b.pdf"),
        ]
        fulltext = [
            (1, 0.7, "content A", {}, "a.pdf"),
            (3, 0.6, "content C", {}, "c.pdf"),
        ]

        fused = retriever._reciprocal_rank_fusion(semantic, fulltext)

        assert len(fused) == 3
        # Doc 1 appears in both lists — should rank first
        assert fused[0][0] == 1
        assert fused[0][1] > fused[1][1]

    def test_rrf_respects_k_constant(self):
        retriever = self._make_retriever(rrf_k=1)
        semantic = [(10, 0.5, "only semantic", {}, "s.pdf")]
        fulltext = [(20, 0.5, "only fulltext", {}, "f.pdf")]

        fused = retriever._reciprocal_rank_fusion(semantic, fulltext)

        # With k=1, rank-1 in each list gives score 1/(1+1) = 0.5 each — tie
        assert len(fused) == 2
        assert fused[0][1] == pytest.approx(0.5)
        assert fused[1][1] == pytest.approx(0.5)

    def test_rrf_empty_lists(self):
        retriever = self._make_retriever()
        fused = retriever._reciprocal_rank_fusion([], [])
        assert fused == []
