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
clear. Demo/testing convenience only; cannot replace counsellor-form
extraction (``leads.features``). Scores from here are provisional.

Unobservable fields stay UNKNOWN, never negative
------------------------------------------------
A chat cannot observe passport status, destination, qualification level,
funding clarity, previous applications, study gap or the counsellor's notes.
``rubric_evidence`` therefore does **not emit them at all** (an earlier
version defaulted passport to "none", which silently forfeited 30% of the
rubric and made "Hot" unreachable). Two consequences, both deliberate:

* **Rule score** = the rubric score over the components a chat can observe
  (``LIVE_OBSERVABLE``: course, intake, English), renormalised to [0, 1].
  The full-form score can only be reached with a counsellor form; a live chat
  that has raised every observable topic scores 0.78, so all three labels are
  reachable (see ``tests/test_live_path.py``).
* **ML input**: any field the chat did not supply is imputed by
  ``leads.hybrid.preprocess_for_prediction`` from the training table shipped
  in ``artifacts/config.json`` (median / mode), so "unknown" means "typical".

Funding clarity is a partial exception: :func:`extract_funding_amount` looks
for an explicit amount (``£15,000`` / ``£20k`` / ``twenty thousand``, ...)
alongside a funding-method phrase (self-fund, savings, parents, a sponsor, an
ability to pay, ...) in a turn's own text, and sets ``funding_clarity`` to a
tier (1/2/3) from the amount. A method mentioned without an amount leaves the
session's current value alone rather than resetting it; a session that never
names an amount this way stays at 0 (unknown), as before. This value is real
evidence for ``leads.rubric.score_row`` and for the ML path (both see the full
``rubric_evidence()`` dict), but is deliberately **not** added to
``LIVE_OBSERVABLE`` below: doing so would move the live rule-score ceiling
documented above, a separate decision.

COUNSELLOR-ONLY FEATURES (deliberately absent from rubric_evidence):
``note_word_count`` / note completeness and ``study_gap_mentioned`` are
form-extraction features -- how fully a counsellor filled the assessment notes
and whether a gap was recorded on the form. A live chat has neither a form nor
a counsellor, so nothing is invented for them (the visitor's own word count is
reported separately as the engagement feature ``session_word_count``).
"""
from __future__ import annotations

import logging
import math
import re
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional

from leads.hybrid import DEFAULT_ALPHA, hybrid_score
from leads.rubric import score_row

_LOG = logging.getLogger(__name__)

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

#: Rubric components (``score_row`` contribution names) a live chat can
#: observe through the topic proxy. The rule score is computed over these only.
LIVE_OBSERVABLE = ("has_course", "has_intake", "english_test")


def _normalise_category(raw: Any) -> str:
    name = str(raw or "General Enquiries")
    return name if name in KNOWN_CATEGORIES else "General Enquiries"


def category_entropy(categories: List[str]) -> float:
    """Shannon entropy in **bits** (log2) of the category distribution.

    Bits, because the synthetic training sessions
    (``leads.personas._category_entropy``) use log2; the ML model was fitted on
    that scale (a nats value would be ~30% too small).
    """
    counts = Counter(_normalise_category(c) for c in categories)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return round(-sum((n / total) * math.log2(n / total)
                      for n in counts.values()), 4)


#: Currency amounts: "£15,000" / "£15000" / "15,000 pounds" / "£20k" / "20k".
#: Case-insensitive (re.IGNORECASE below); commas are stripped before parsing.
_AMOUNT_NUM = r"\d[\d,]*(?:\.\d+)?"
_AMOUNT_RE = re.compile(
    rf"£\s*(?P<a1>{_AMOUNT_NUM})\s*(?P<k1>k)?\b"
    rf"|(?P<a2>{_AMOUNT_NUM})\s*k\b"
    rf"|(?P<a3>{_AMOUNT_NUM})\s*pounds?\b",
    re.IGNORECASE,
)

#: "twenty thousand" / "fifteen hundred": counted only when a number word is
#: immediately followed by a scale word, so a bare "one" (as in "a one-year
#: masters") or "three" (as in "three instalments") is never mistaken for an
#: amount.
_WORD_VALUES = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_WORD_SCALES = {"hundred": 100, "thousand": 1000}


def _value_from_match(m: "re.Match[str]") -> Optional[float]:
    if m.group("a1") is not None:
        v = float(m.group("a1").replace(",", ""))
        return v * 1000 if m.group("k1") else v
    if m.group("a2") is not None:
        return float(m.group("a2").replace(",", "")) * 1000
    if m.group("a3") is not None:
        return float(m.group("a3").replace(",", ""))
    return None


def _word_amount(text: str) -> Optional[float]:
    """First "<number words> <hundred|thousand>" phrase in *text*, if any."""
    words = re.findall(r"[a-z]+", text.lower())
    i = 0
    while i < len(words):
        if words[i] not in _WORD_VALUES:
            i += 1
            continue
        j, total = i, 0
        while j < len(words) and words[j] in _WORD_VALUES:
            total += _WORD_VALUES[words[j]]
            j += 1
        if j < len(words) and words[j] in _WORD_SCALES:
            return float(total * _WORD_SCALES[words[j]])
        i = j
    return None


def extract_funding_amount(text: str) -> Optional[float]:
    """The funding amount named in *text*, or ``None``.

    Matches ``£15,000`` / ``£15000`` / ``15,000 pounds`` / ``£20k`` / ``20k`` /
    a word amount like "twenty thousand" -- case-insensitive, commas stripped.
    A range ("£15,000-£20,000") and an "up to X" phrase both resolve the same
    way: of every amount named, the lowest is returned (a range's lower bound;
    an "up to X" phrase's only number, trivially).
    """
    if not text:
        return None
    amounts = [v for v in (_value_from_match(m) for m in _AMOUNT_RE.finditer(text))
               if v is not None]
    if amounts:
        return min(amounts)
    return _word_amount(text)


#: Phrases that name a funding *method* (self-funding, parents, savings, a
#: sponsor, an ability/budget to pay, or its absence). Stemmed loosely ("sav"
#: catches saved/saving/savings; "parent" catches parents/parent's) so ordinary
#: inflections are not missed; matched as a case-insensitive substring.
_FUNDING_METHOD_MARKERS = (
    "self fund",        # self-fund / self fund / self-funded / self-funding
    "sav",              # savings / saved / saving
    "parent",           # parents pay / parents have / parent's
    "can pay",
    "budget",
    "sponsor",
    "source of fund",   # "no source of funds"
)


def _mentions_funding_method(text: str) -> bool:
    t = (text or "").lower().replace("-", " ")
    return any(marker in t for marker in _FUNDING_METHOD_MARKERS)


def _funding_tier(amount: float) -> int:
    """amount -> the funding_clarity level score_row() already understands
    (1/2/3, i.e. leads.features.FUNDING_UNCLEAR/PARTIAL/CLEAR)."""
    if amount < 10_000:
        return 1
    if amount < 20_000:
        return 2
    return 3


def empty_evidence() -> Dict[str, Any]:
    """Baseline evidence row: only what a chat can observe, all absent."""
    return {
        "has_english_test": 0,
        "english_band": 0.0,
        "funding_method_present": 0,
        "funding_clarity": 0,
        "has_course": 0,
        "has_intake": 0,
    }


def live_rule_breakdown(evidence: Dict[str, Any]) -> Dict[str, Any]:
    """The arithmetic behind the live rule score.

    ``lines`` are the ``LIVE_OBSERVABLE`` rubric components (feature,
    contribution, weight); ``raw`` is their summed contribution, ``mass`` their
    summed weight, and ``score = raw / mass``. The dashboard panel and
    :func:`rule_score_from_evidence` both come from here, so what is shown
    always adds up to the score beside it (contributions of 0.050 + 0.040 + 0.000
    = 0.090 over a mass of 0.180 is 0.500, not 0.090).
    """
    lines = []
    got = mass = 0.0
    for c in score_row(dict(evidence)).contributions:
        if c["feature"] in LIVE_OBSERVABLE:
            lines.append({"feature": c["feature"],
                          "contribution": float(c["contribution"]),
                          "weight": float(c["weight"]),
                          "detail": c.get("detail")})
            got += float(c["contribution"])
            mass += float(c["weight"])
    score = float(got / mass) if mass else 0.0
    _LOG.debug(
        "live_rule_breakdown: raw=%.4f mass=%.4f score=%.4f (score = raw / mass; "
        "no clamp) components=%s", got, mass, score,
        ", ".join("%s=%+.3f/%.3f" % (l["feature"], l["contribution"], l["weight"])
                  for l in lines))
    return {"lines": lines, "raw": got, "mass": mass, "score": score}


def rule_score_from_evidence(evidence: Dict[str, Any]) -> float:
    """Rule score in [0, 1]: the rubric over ``LIVE_OBSERVABLE``, renormalised.

    ``score_row`` weights the full counsellor form; dividing by the weight mass
    of the components a chat can observe keeps the score on the same [0, 1]
    scale the label cuts assume (unobserved components are excluded, not
    scored as zero). See :func:`live_rule_breakdown`.
    """
    return live_rule_breakdown(evidence)["score"]


def hybrid_for_session(rule: float, ml_proba=None,
                       alpha: float = DEFAULT_ALPHA) -> float:
    """Hybrid score; rule-only when ``ml_proba`` is None."""
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
        """Engagement metrics for the session so far.

        ``avg_delay_s`` is ``None`` until two turns exist: with fewer there is
        no delay to observe, and ``0.0`` would read as "instant replies" (below
        every synthetic training range). ``leads.hybrid`` imputes ``None`` with
        the training median.
        """
        n = len(self._turns)
        words = sum(len(t["user_msg"].split()) for t in self._turns)
        delays: List[float] = []
        for prev, cur in zip(self._turns, self._turns[1:]):
            try:
                delays.append(max(0.0, (cur["ts"] - prev["ts"]).total_seconds()))
            except (TypeError, AttributeError):
                continue
        avg_delay: Optional[float] = (
            round(sum(delays) / len(delays), 2) if delays else None)
        visa = any(t["category"] == "Visa & Immigration" for t in self._turns)
        return {
            "message_count": n,
            "avg_delay_s": avg_delay,
            "session_word_count": words,
            "question_category_entropy": category_entropy(
                [t["category"] for t in self._turns]),
            "visa_intent_mentioned": 1 if visa else 0,
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
        """Map seen categories onto the rubric fields a chat can observe.

        Proxy mapping (see module docstring -- demo approximation):
          Visa & Immigration -> ``session_flags.visa_intent_mentioned``
          Fees & Funding / Scholarships -> ``funding_method_present=1``
            (interest is not proof of funds; see ``funding_clarity`` below)
          English Language -> ``has_english_test=1`` (proxy for readiness;
            band stays 0.0 = present but unparsed)
          Entry Requirements -> ``has_course=1``
          Application Process -> ``has_intake=1``
          A turn naming both an amount and a funding method (see
            ``extract_funding_amount`` / ``_mentions_funding_method``) sets
            ``funding_clarity`` to that turn's tier; a later qualifying turn
            overwrites it, a method-only turn leaves it as is.
        Passport, destination, qualification, previous applications, study gap
        and note length are unobservable in chat and are **absent** from the
        row (unknown, not negative).
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
        for t in self._turns:
            if not _mentions_funding_method(t["user_msg"]):
                continue
            amount = extract_funding_amount(t["user_msg"])
            if amount is not None:
                ev["funding_clarity"] = _funding_tier(amount)
            # else: a funding method named without an amount -- leave the
            # session's current funding_clarity as is (never reset to unknown).
        out = dict(ev)
        out["session_flags"] = flags
        return out

    def rule_score(self) -> float:
        """Rule score in [0, 1] for the current evidence.

        An empty session scores 0.0: with no turns there is no evidence,
        so no rubric credit applies.
        """
        if not self._turns:
            return 0.0
        return rule_score_from_evidence(self.rubric_evidence())

    def hybrid(self, ml_proba=None, alpha: float = DEFAULT_ALPHA) -> float:
        """Hybrid score; rule-only unless ``ml_proba`` is supplied.

        Empty sessions score 0.0 regardless of any supplied proba: there
        are no features to score.
        """
        if not self._turns:
            return 0.0
        return hybrid_for_session(self.rule_score(), ml_proba, alpha=alpha)
