"""Multi-intent guard: is a query really several questions at once?

Lives here, not in ``dashboard/app.py``, so that ``chatbot.responder.respond``
can report **which** keyword groups matched in its debug line. The guard still
does not run inside ``respond``: the dashboard calls :func:`is_multi_intent` and
passes ``force_abstain=True``.

How it decides: the query is matched against five topical keyword groups
(passport, English test, KCSE/grades, programme, intake dates). A query that
touches **more than** ``THRESHOLD`` (2) groups is treated as compound and
abstains + escalates, whatever its retrieval confidence. An institution name is
deliberately *not* a group: it scopes a question ("... at RGU"), it does not add
one, so naming an institution never changes the verdict.
"""
from __future__ import annotations

import re
from typing import List, Tuple

#: Keyword groups; a query "hits" a group when any keyword occurs as a whole
#: word/phrase. The index of a group is its identity in debug output (g0..g4).
INTENT_GROUPS = (
    ("passport",),
    ("ielts", "english test", "english-test"),
    ("kcse", "grade", "mean grade", "b+", "c+", "gpa"),
    ("data science", "nursing", "business", "course", "degree",
     "masters", "programme", "program", "subject"),
    ("intake", "september", "january", "deadline", "when can i start",
     "when do i apply"),
)

#: A query hitting more than this many groups is treated as multi-intent.
THRESHOLD = 2

#: Phrases that contain an intent keyword without expressing that intent.
NOT_INTENT = re.compile(r"\bof course\b")


def kw_in(query: str, keyword: str) -> bool:
    """Whole-word / whole-phrase match (``grade`` must not match ``upgrade``)."""
    return re.search(r"(?<!\w)" + re.escape(keyword) + r"(?!\w)", query) is not None


def matched_intent_groups(query: str) -> List[Tuple[int, List[str]]]:
    """``[(group_index, [keywords that matched, in group order]), ...]``."""
    q = NOT_INTENT.sub(" ", query.lower())
    out: List[Tuple[int, List[str]]] = []
    for i, group in enumerate(INTENT_GROUPS):
        hit = [k for k in group if kw_in(q, k)]
        if hit:
            out.append((i, hit))
    return out


def is_multi_intent(query: str) -> bool:
    """True when the query hits more than ``THRESHOLD`` distinct groups."""
    return len(matched_intent_groups(query)) > THRESHOLD
