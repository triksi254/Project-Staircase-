"""Tests for chatbot.embeddings that need NO ML dependencies.

Only the dependency-free half of the module is exercised here (fingerprint,
graceful degradation, cache-path determinism). The model-backed behaviour is
in tests/test_sbert_retriever.py, which skips when the ML stack is absent.
"""
from pathlib import Path

from chatbot.embeddings import SbertRetriever, available, corpus_fingerprint
from chatbot.retriever import FaqEntry, load_corpus


def _entries():
    return [FaqEntry(question="What are the entry requirements?", answer="A 2:2.",
                     keywords=["entry"], index=0),
            FaqEntry(question="How much are the fees?", answer="GBP 15000.",
                     keywords=["fees"], index=1)]


def test_available_reports_bool():
    assert isinstance(available(), bool)


def test_corpus_fingerprint_deterministic_and_content_sensitive():
    a, b = _entries(), _entries()
    assert corpus_fingerprint(a) == corpus_fingerprint(b)
    # editing an answer must change the fingerprint (cache invalidation key)
    b[1].answer = "GBP 16000."
    assert corpus_fingerprint(a) != corpus_fingerprint(b)


def test_corpus_fingerprint_depends_on_model_name():
    assert (corpus_fingerprint(_entries(), "all-MiniLM-L6-v2")
            != corpus_fingerprint(_entries(), "all-mpnet-base-v2"))


def test_empty_corpus_needs_no_model():
    """An empty corpus must not load a model or build a matrix."""
    retriever = SbertRetriever([], cache=False)
    assert retriever.entries == []
    assert retriever.search("anything") == []
    assert retriever.confidence("anything") == 0.0
    assert retriever._model is None


def test_blank_query_short_circuits_before_encoding():
    """A blank query returns [] even with no model loaded."""
    retriever = SbertRetriever(entries=[], cache=False)
    assert retriever.search("   ") == []


def test_cache_paths_are_stable_and_model_specific(tmp_path):
    one = SbertRetriever([], cache=False, cache_dir=tmp_path)
    two = SbertRetriever([], cache=False, cache_dir=tmp_path)
    assert one._cache_paths() == two._cache_paths()
    other = SbertRetriever([], cache=False, cache_dir=tmp_path,
                           model_name="all-mpnet-base-v2")
    assert other._cache_paths() != one._cache_paths()
    assert one._cache_paths()[0].parent == Path(tmp_path)


def test_real_corpus_fingerprint_is_content_hash():
    entries = load_corpus()
    assert len(entries) == 98
    fp = corpus_fingerprint(entries)
    assert len(fp) == 16 and fp.isalnum()
    # stable across reloads of the same corpus
    assert corpus_fingerprint(load_corpus()) == fp

# --------------------------------------------------------------------------- #
# model loading prefers the local HF cache (no hub request / auth warning)
# --------------------------------------------------------------------------- #
def _bare_retriever():
    from chatbot.embeddings import SbertRetriever
    r = SbertRetriever.__new__(SbertRetriever)
    r.model_name, r.device = "all-MiniLM-L6-v2", "cpu"
    return r


def test_model_is_loaded_from_the_local_cache_first(monkeypatch):
    import sentence_transformers
    calls = []

    class _Model:
        def __init__(self, name, device="cpu", **kw):
            calls.append(kw)

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _Model)
    _bare_retriever()._load_model()
    assert calls == [{"local_files_only": True}]


def test_model_falls_back_to_the_network_when_not_cached(monkeypatch):
    import sentence_transformers
    calls = []

    class _Model:
        def __init__(self, name, device="cpu", **kw):
            calls.append(kw)
            if kw.get("local_files_only"):
                raise OSError("not in the local cache")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _Model)
    _bare_retriever()._load_model()
    assert calls == [{"local_files_only": True}, {}]
