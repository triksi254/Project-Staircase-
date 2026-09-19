"""SBERT-backed retrieval tests; skipped when the ML stack is not installed.

Deliberately isolated from tests/test_embeddings.py so that
``pip install -r requirements.txt`` alone still yields a fully green suite.
"""
import pytest

from chatbot.embeddings import SbertRetriever, available
from chatbot.retriever import FaqEntry, load_corpus

pytestmark = pytest.mark.skipif(
    not available(), reason="sentence-transformers not installed "
                            "(pip install -r requirements-ml.txt)")


@pytest.fixture(scope="module")
def retriever():
    return SbertRetriever(load_corpus(), cache=False)


def test_search_returns_entries_and_cosines(retriever):
    hits = retriever.search("Do I need IELTS if I did KCSE English?", top_k=3)
    assert len(hits) == 3
    for entry, score in hits:
        assert isinstance(entry, FaqEntry)
        assert 0.0 <= score <= 1.0


def test_scores_are_monotonically_non_increasing(retriever):
    hits = retriever.search("how much is accommodation per week at Aston",
                            top_k=5)
    scores = [s for _e, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_semantic_match_beats_keyword_overlap(retriever):
    """The point of SBERT: a paraphrase with little lexical overlap still wins."""
    hits = retriever.search("is KCSE English accepted instead of IELTS", top_k=3)
    top = hits[0][0]
    assert "English" in top.category or "IELTS" in top.question.upper() \
        or "KCSE English" in top.question


def test_sbert_retriever_reports_raw_cosine_hook(retriever):
    """retrieval_eval reads this hook to state H1 on pure cosine."""
    value = retriever.raw_top1_cosine("Can I work while studying abroad?")
    assert 0.0 <= value <= 1.0
    assert value == retriever.confidence("Can I work while studying abroad?")


def test_index_contract_holds(retriever):
    """Gold ids address the raw JSON corpus; SBERT must preserve entry.index."""
    for pos, entry in enumerate(retriever.entries):
        assert entry.index == pos


def test_cache_round_trip_reproduces_scores(tmp_path):
    entries = load_corpus()[:20]
    cold = SbertRetriever(entries, cache_dir=tmp_path)
    before = cold.search("What are the entry requirements?", top_k=3)
    npy, meta = cold._cache_paths()
    assert npy.is_file() and meta.is_file()
    warm = SbertRetriever(entries, cache_dir=tmp_path)  # loads from cache
    after = warm.search("What are the entry requirements?", top_k=3)
    assert [e.index for e, _s in before] == [e.index for e, _s in after]
    assert [s for _e, s in before] == [s for _e, s in after]


def test_cache_invalidated_when_corpus_changes(tmp_path):
    entries = load_corpus()[:20]
    first = SbertRetriever(entries, cache_dir=tmp_path)
    npy, _meta = first._cache_paths()
    edited = list(entries)
    edited = [FaqEntry(question=edited[0].question, answer="Completely new text.",
                       keywords=[], index=0)] + edited[1:]
    second = SbertRetriever(edited, cache_dir=tmp_path)
    assert second._cache_paths()[0] != npy  # different fingerprint -> new file


def test_missing_cache_dir_is_created(tmp_path):
    target = tmp_path / "nested" / "embeds"
    SbertRetriever(load_corpus()[:5], cache_dir=target)
    assert target.is_dir()