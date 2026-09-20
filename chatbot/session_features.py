"""Live-session features for the counsellor test platform.

Derives rubric-relevant engagement metrics and evidence flags from an
in-progress conversation so the dashboard can score a lead *while the chat
is happening*, instead of waiting for a counsellor form.

APPROXIMATION WARNING (read before citing numbers from this module):
the mapping in ``SessionTracker.rubric_evidence`` infers counsellor-form
fields (``has_english_test``, ``has_course``, ...) from *which FAQ
categories the visitor asked about*. Topic interest is NOT the same thing
as holding the document: asking about IELTS does not mean the visitor has
an IELTS certificate, and asking about fees does not mean funding is
clear. Anything unseen defaults to the "unknown/none" bucket, never to a
positive. Demo/testing convenience only; cannot replace counsellor-form
extraction (``leads.features``). Scores from here are rule-only /
provisional by definition.

 COUNSELLOR-ONLY FEATURES (deliberately absent from rubric_evidence):
 note_completeness and study_gap are form-extraction features -- the
 former scores how fully a counsellor filled the assessment notes, the
 latter whether a gap was recorded on the form. A live chat has neither
 a form nor a counsellor, so this module does not invent values for them:
 note_completeness keeps its score_row default, and
 ``study_gap_mentioned`` is emitted only once the visitor has actually
 sent a turn. An empty session has no gap evidence either way, so it must
 not collect the rubric's "no gap" credit before the first message.
"""
from __future__ import annotations

import math
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

from leads.features import FUNDING_UNKNOWN, PASSPORT_NONE
from leads.hybrid import hybrid_score
from leads.rubric import score_row

#: Categories the live classifier/responder may emit. Anything outside this
#: set is counted under "General Enquiries" for entropy purposes.
KNOWN_CATEGORIES = (
    "Program Details",
    "Fees & Funding",
    "Entry Requirements",
    "Accommodation",
    "Scholarships",
    "English Language",
    "Visa & Immigration",
    "Application Process",
    "General Enquiries",
)


def _normalise_category(raw: Any) -> str:
    name = str(raw or "General Enquiries")
    return name if name in KNOWN_CATEGORIES else "General Enquiries"


def category_entropy(categories: List[str]) -> float:
    """Shannon entropy (nats) of the category distribution."""
    counts = Counter(_normalise_category(c) for c in categories)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return round(-sum((n / total) * math.log(n / total)
                      for n in counts.values()), 4)


def empty_evidence() -> Dict[str, Any]:
    """Baseline evidence row: every signal unknown/absent, never positive."""
    return {
        "passport_status": PASSPORT_NONE,
        "has_english_test": 0,
        "english_band": 0.0,
        "funding_method_present": 0,
        "funding_clarity": FUNDING_UNKNOWN,
        "destination_uk": 0,
        "has_course": 0,
        "has_intake": 0,
        "qual_level": 0,
        "study_gap_mentioned": 0,
        "previous_application_mentioned": 0,
        "note_word_count": 0,
    }


def rule_score_from_evidence(evidence: Dict[str, Any]) -> float:
    """Rule score in [0, 1] for an evidence row via ``leads.rubric``."""
    return float(score_row(dict(evidence)).score)


def hybrid_for_session(rule: float, ml_proba=None,
                       alpha: float = 0.5) -> float:
    """Hybrid score; rule-only when ``ml_proba`` is None (dashboard default)."""
    return float(hybrid_score(rule, ml_proba, alpha=alpha))


class SessionTracker:
    """Accumulate one lead's conversation turns into scoring features."""

    def __init__(self, lead_id: str, started_at: datetime) -> None:
        self.lead_id = lead_id
        self.started_at = started_at
        self._turns: List[Dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._turns)

    def add_turn(self, user_msg: str, bot_resp: dict, ts: datetime) -> None:
        """Record one exchange. ``bot_resp`` is the ``respond()`` dict."""
        bot_resp = bot_resp or {}
        self._turns.append({
            "user_msg": user_msg or "",
            "category": _normalise_category(bot_resp.get("category")),
            "confidence": float(bot_resp.get("confidence", 0.0) or 0.0),
            "ts": ts,
        })

    def features(self) -> Dict[str, Any]:
        """Engagement metrics for the session so far."""
        n = len(self._turns)
        words = sum(len(t["user_msg"].split()) for t in self._turns)
        delays: List[float] = []
        for prev, cur in zip(self._turns, self._turns[1:]):
            try:
                delays.append(max(0.0, (cur["ts"] - prev["ts"]).total_seconds()))
            except (TypeError, AttributeError):
                continue
        avg_delay = round(sum(delays) / len(delays), 2) if delays else 0.0
        return {
            "message_count": n,
            "avg_delay_s": avg_delay,
            "session_word_count": words,
            "question_category_entropy": category_entropy(
                [t["category"] for t in self._turns]),
            # In-session view only; cross-session identity is not tracked.
            "returning_session": 0,
        }

    def category_counts(self) -> Dict[str, int]:
        """Turn count per (normalised) category, in first-seen order."""
        counts: Dict[str, int] = {}
        for t in self._turns:
            counts[t["category"]] = counts.get(t["category"], 0) + 1
        return counts

    def rubric_evidence(self) -> Dict[str, Any]:
        """Map seen categories onto rubric fields; unseen stays non-positive.

        Proxy mapping (see module docstring -- demo approximation):
          Visa & Immigration -> ``visa_intent_mentioned=True`` (session flag)
          Fees & Funding / Scholarships -> ``funding_method_present=True``
            (clarity stays UNKNOWN; interest is not proof of funds)
          English Language -> ``has_english_test=1`` (proxy for readiness;
            band stays 0.0 = present but unparsed)
          Entry Requirements -> ``has_course=True``
          Application Process -> ``has_intake=True``
        ``destination_uk`` stays 0 (topic interest does not establish
        destination). Passport/qualifications are unobservable in chat and
        stay at their unknown defaults. ``study_gap_mentioned`` is emitted as
        0 ("no gap recorded") only once at least one turn exists; an empty
        session omits the key so a silent lead cannot collect the rubric's
        "no gap" credit.
        """
        ev = empty_evidence()
        seen = set(self.category_counts())
        flags: Dict[str, Any] = {
            "visa_intent_mentioned": ("Visa & Immigration" in seen)}
        if "Fees & Funding" in seen or "Scholarships" in seen:
            ev["funding_method_present"] = 1
            flags["funding_method_present"] = True
        else:
            flags["funding_method_present"] = False
        if "English Language" in seen:
            ev["has_english_test"] = 1
        if "Entry Requirements" in seen:
            ev["has_course"] = 1
        if "Application Process" in seen:
            ev["has_intake"] = 1
        ev["note_word_count"] = sum(len(t["user_msg"].split())
                                    for t in self._turns)
        out = dict(ev)
        if not self._turns:
            # Silent lead: no gap evidence either way. Omitting the key keeps
            # the rubric's "no gap" credit out of an untouched session.
            out.pop("study_gap_mentioned", None)
        out["session_flags"] = flags
        return out

    def rule_score(self) -> float:
        """Rule score in [0, 1] for the current evidence.

        An empty session scores 0.0: with no turns there is no evidence,
        so no rubric credit (including the "no gap" credit) applies.
        """
        if not self._turns:
            return 0.0
        return rule_score_from_evidence(self.rubric_evidence())

    def hybrid(self, ml_proba=None, alpha: float = 0.5) -> float:
        """Hybrid score; rule-only unless ``ml_proba`` is supplied.

        Empty sessions score 0.0 regardless of any supplied proba: there
        are no features to score.
        """
        if not self._turns:
            return 0.0
        return hybrid_for_session(self.rule_score(), ml_proba, alpha=alpha)
