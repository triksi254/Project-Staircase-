"""funding_structural_test - a second, differently-phrased funding session.

DIAGNOSIS ONLY: nothing here changes behaviour. LEAD-20260921-0002 (six funding
turns, ``final_rule_score: 0.0``) suggested funding never earns rule-score
credit through chat. This script re-runs the claim with different phrasing
(explicit amounts and years, "£15,000", "£30,000 saved", "three instalments",
"total cost") to check whether the earlier result was about that session's
wording or is structural: ``chatbot.session_features.rubric_evidence()`` never
emits ``funding_clarity`` at all (see its docstring, "unobservable fields stay
UNKNOWN"), and the field it does emit, ``funding_method_present``, is not a
rubric feature (``leads.rubric.WEIGHTS`` has no such key) and is outside
``LIVE_OBSERVABLE``, so ``leads.rubric.score_row`` never reads it either way.

Usage:
    python scripts/funding_structural_test.py [--out artifacts/funding_structural_test.txt]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / "artifacts" / "funding_structural_test.txt"

QUERIES = (
    "I can pay £15,000 per year towards tuition",
    "My parents have £30,000 saved for my education",
    "I'll need a £10,000 scholarship to make it work",
    "Can I pay fees in three instalments over the year?",
    "How much is tuition at Aston for international students?",
    "What is the total cost of a one-year masters at BCU?",
)


def capture(retriever=None):
    """One record per turn, accumulated on a single tracker (one session)."""
    import dashboard.app as app
    from chatbot.session_features import SessionTracker
    from leads.rubric import score_row

    retriever = retriever if retriever is not None else app._get_retriever()
    tracker = SessionTracker("FUNDING-STRUCT-2", datetime.now(timezone.utc))
    records = []
    for n, query in enumerate(QUERIES, 1):
        out = app._answer_query(query, tracker, retriever)
        ev = tracker.rubric_evidence()
        funding_row = score_row(dict(ev))
        funding_line = next(c for c in funding_row.contributions
                            if c["feature"] == "funding_clarity")
        records.append({
            "n": n, "query": query,
            "category": out["category"],
            "confidence": out["confidence"],
            "abstained": out["abstained"],
            "funding_method_present": ev["funding_method_present"],
            "funding_clarity_present_in_evidence": "funding_clarity" in ev,
            "funding_clarity_contribution": funding_line["contribution"],
            "funding_clarity_weight": funding_line["weight"],
            "funding_clarity_detail": funding_line["detail"],
            "rule_score": tracker.rule_score(),
            "evidence": ev,
        })
    return records


def render(records):
    """The report text for ``records`` (pure; no retrieval)."""
    rows = [(str(r["n"]), r["category"], str(r["funding_method_present"]),
             "absent" if not r["funding_clarity_present_in_evidence"] else "present",
             "%.4f" % r["funding_clarity_contribution"], "%.4f" % r["rule_score"])
            for r in records]
    head = ("#", "category", "funding_method_present", "funding_clarity key",
            "funding contribution", "rule_score")
    widths = [max(len(head[i]), *(len(row[i]) for row in rows)) for i in range(len(head))]

    def line(cells):
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()

    ever_above_zero = any(r["funding_clarity_contribution"] > 0 for r in records)
    ever_key_present = any(r["funding_clarity_present_in_evidence"] for r in records)

    out = [
        "funding_structural_test - a second funding session, varied phrasing",
        "DIAGNOSIS ONLY: no fixes applied.",
        "Path: dashboard/app.py::_answer_query, one tracker for all 6 turns (one session).",
        "Reproduce: python scripts/funding_structural_test.py",
        "",
        "Compare: LEAD-20260921-0002 (6 funding turns, plain phrasing) ended at",
        "final_rule_score = 0.0 despite discussing tuition, scholarships and instalments.",
        "",
        "SUMMARY",
        line(head), line(["-" * w for w in widths]),
    ]
    out += [line(row) for row in rows]
    out += [
        "",
        "funding_clarity ever present in rubric_evidence(): %s" % ever_key_present,
        "funding_clarity contribution ever > 0.0: %s" % ever_above_zero,
        "",
        "DETAIL",
    ]
    for r in records:
        out += [
            "",
            "[%d] %s" % (r["n"], r["query"]),
            "    category assigned (this turn)      : %s (confidence %.4f, %s)" % (
                r["category"], r["confidence"], "abstained" if r["abstained"] else "answered"),
            "    funding_method_present (cumulative) : %s" % r["funding_method_present"],
            "    funding_clarity in evidence row     : %s" % (
                "present" if r["funding_clarity_present_in_evidence"] else "absent (unobservable; see module docstring)"),
            "    funding_clarity rubric contribution : %.4f / %.4f weight  (detail: %s)" % (
                r["funding_clarity_contribution"], r["funding_clarity_weight"], r["funding_clarity_detail"]),
            "    live rule_score (cumulative)        : %.4f" % r["rule_score"],
            "    rubric_evidence() dict (cumulative) : %s" % r["evidence"],
        ]
    out += [
        "",
        "VERDICT",
        ("funding_clarity never left 0.0 in this session either: rubric_evidence() "
         "does not set the key regardless of phrasing (score_row() then defaults it "
         "to FUNDING_UNKNOWN = 0.0 contribution)." if not ever_above_zero and not ever_key_present
         else "funding_clarity was observed above 0.0 or the key appeared; the earlier "
              "session's result does not generalise as claimed -- see DETAIL above."),
        "funding_method_present is set (Fees & Funding / Scholarships turns) but is not a",
        "rubric feature (leads.rubric.WEIGHTS has no such key) and is outside",
        "chatbot.session_features.LIVE_OBSERVABLE, so score_row() never reads it and the",
        "live rule score never includes it either way. This is structural, not phrasing-",
        "dependent: no live chat can move funding_clarity, whatever the wording.",
    ]
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="funding_structural_test (diagnosis only)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    text = render(capture())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))
    print("Wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
