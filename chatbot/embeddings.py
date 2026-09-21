"""Sentence-BERT semantic retrieval over the grounded FAQ corpus.

Same interface as ``chatbot.retriever.Retriever`` (``search`` / ``confidence``),
so ``evaluation`` can run both retrievers on identical gold sets. Similarity is
the cosine of L2-normalised sentence embeddings, i.e. a plain inner product;
unlike the TF-IDF path there is no keyword bonus, so the scores are directly
comparable to the proposal's "cosine similarity" wording.

Why not FAISS: at 98 documents an exact brute-force dot product takes
microseconds and returns the *true* nearest neighbours, whereas an
approximate index adds a dependency and a recall caveat for no latency
benefit. The interface is unchanged, so dropping in ``faiss.IndexFlatIP``
later is a one-line swap inside ``_build_index``.

Optional by design: ``sentence-transformers`` and ``torch`` are deliberately
NOT in ``requirements.txt`` (see ``requirements-ml.txt``). Nothing imports
this module at load time; constructing ``SbertRetriever`` without them raises
an ImportError with an actionable message, and the TF-IDF path is unaffected.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBED_DIR = PROJECT_ROOT / "data"
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def available() -> bool:
    """True when sentence-transformers can be imported."""
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        return False
    return True


def corpus_fingerprint(entries, model_name: str = DEFAULT_MODEL) -> str:
    """Stable id for (model, corpus contents) - cache invalidation key.

    Hashes every entry's *document* text, not just the count, so editing an
    answer or adding an FAQ invalidates the cached embedding matrix.
    """
    digest = hashlib.sha256()
    digest.update(model_name.encode("utf-8"))
    for entry in entries:
        digest.update(("%d|%s" % (entry.index, entry.document)).encode("utf-8"))
    return digest.hexdigest()[:16]


class SbertRetriever:
    """Rank corpus entries by Sentence-BERT cosine. ``search`` returns (entry, score)."""

    #: Backend id used by the dashboard / transcripts to pick the abstention gate.
    backend = "sbert"

    def __init__(self, entries, model_name: str = DEFAULT_MODEL,
                 device: str = "cpu", cache: bool = True,
                 batch_size: int = 32, cache_dir=None) -> None:
        self.entries = list(entries)
        self.model_name = model_name
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir is not None else EMBED_DIR
        self._model: Any = None
        self._doc_matrix: Any = None
        self._fingerprint = corpus_fingerprint(self.entries, model_name)
        if self.entries:
            self._build_index(cache=cache, batch_size=batch_size)

    @property
    def model(self):
        """Lazily load the SentenceTransformer (so import cost is deferred)."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "SbertRetriever requires sentence-transformers and torch. "
                "Install with: pip install -r requirements-ml.txt") from exc
        # Prefer the local HuggingFace cache: no hub request (so no "unauthenticated
        # requests to the HF Hub" warning on every start, and no network latency).
        # Only when the model is not cached yet do we fall back to downloading it.
        try:
            return SentenceTransformer(self.model_name, device=self.device,
                                       local_files_only=True)
        except (OSError, ValueError):
            logger.info("embeddings: %s not in the local cache - downloading",
                        self.model_name)
            return SentenceTransformer(self.model_name, device=self.device)

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def _cache_paths(self) -> Tuple[Path, Path]:
        stem = "embeddings_%s_%s" % (self.model_name.replace("/", "_"),
                                     self._fingerprint)
        return (self.cache_dir / (stem + ".npy"),
                self.cache_dir / (stem + ".json"))

    def _build_index(self, cache: bool = True, batch_size: int = 32) -> None:
        """Encode the corpus (or load a valid cache) into a normalised matrix."""
        import numpy as np

        npy, meta = self._cache_paths()
        if cache and npy.is_file() and meta.is_file():
            loaded = self._load_cache(npy, meta)
            if loaded is not None:
                self._doc_matrix = loaded
                return

        docs = [e.document for e in self.entries]
        arr = self.model.encode(docs, batch_size=batch_size,
                                convert_to_numpy=True,
                                normalize_embeddings=True,
                                show_progress_bar=False)
        self._doc_matrix = np.asarray(arr, dtype="float32")
        if cache:
            self._save_cache(npy, meta)

    def _load_cache(self, npy: Path, meta: Path):
        """Return the cached matrix when it matches this corpus, else None."""
        import numpy as np

        try:
            info = json.loads(meta.read_text(encoding="utf-8"))
            if (info.get("fingerprint") != self._fingerprint
                    or info.get("model") != self.model_name
                    or info.get("n") != len(self.entries)):
                return None
            matrix = np.load(npy)
        except (OSError, ValueError) as exc:
            logger.warning("embeddings: cache unusable (%s) - re-encoding", exc)
            return None
        if matrix.shape[0] != len(self.entries):
            return None
        return np.asarray(matrix, dtype="float32")

    def _save_cache(self, npy: Path, meta: Path) -> None:
        """Persist the embedding matrix; failure is a warning, never fatal."""
        import numpy as np

        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(npy, self._doc_matrix)
            meta.write_text(json.dumps({
                "model": self.model_name,
                "fingerprint": self._fingerprint,
                "n": len(self.entries),
                "dim": int(self._doc_matrix.shape[1]),
                "normalized": True,
            }, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            logger.warning("embeddings: caching failed (%s)", exc)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[Any, float]]:
        """Top-k (entry, cosine in [0, 1]); [] for empty query/corpus."""
        if not query or not query.strip() or not self.entries:
            return []
        if self._doc_matrix is None:
            return []
        import numpy as np

        q = self.model.encode([query], convert_to_numpy=True,
                              normalize_embeddings=True,
                              show_progress_bar=False)
        sims = self._doc_matrix @ np.asarray(q[0], dtype="float32")
        k = min(max(1, top_k), int(sims.shape[0]))
        order = np.argsort(-sims)[:k]
        return [(self.entries[int(i)],
                 round(float(max(0.0, min(1.0, sims[int(i)]))), 4))
                for i in order]

    def confidence(self, query: str) -> float:
        """Top-1 cosine or 0.0 when nothing matches."""
        hits = self.search(query, top_k=1)
        return hits[0][1] if hits else 0.0

    def raw_top1_cosine(self, query: str) -> float:
        """Hook used by evaluation.retrieval_eval for H1's raw-cosine reporting.

        For this retriever the reported score already *is* the raw cosine (no
        keyword bonus), so this is an alias - but exposing it keeps H1 stated
        on the same quantity for both retrievers.
        """
        return self.confidence(query)