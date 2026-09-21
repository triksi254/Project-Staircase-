"""guard_fix_verification - how 8 test queries behave after ``guard_institution_fix``.

CAPTURE ONLY: nothing here changes behaviour, and the report does not judge
whether an outcome is right. Each query goes through the dashboard's own
``dashboard.app._answer_query`` (real retriever, its gate, the multi-intent
guard, a fresh tracker), so the decision is what a visitor's first chat turn
gets. The log line quoted per query is ``respond()``'s DEBUG line.

Usage:
    python scripts/guard_fix_verification.py [--out artifacts/guard_fix_verification.txt]
"""
from __future__ import annotations

import argparse
import contextlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / "artifacts" / "guard_fix_verification.txt"

QUERIES = (
    "Do I need IELTS to study at Aston if I did KCSE English?",
    "What are the entry requirements for University of Birmingham?",
    "I need to know about IELTS, scholarships, and accommodation",
    "Which university should I study at in the UK?",
    "I have a Kenyan passport. What do I need for a masters in the UK?",
    "I have a Kenyan passport, IELTS 6.5, and want to study Data Science",
    "How much does accommodation cost at Aston University?",
    "What IELTS score do I need for a masters at RGU?",
)


def _groups_repr(groups):
    """``[g1:ielts, g3:masters]`` (``[]`` when nothing matched)."""
    return "[" + ", ".join("g%d:%s" % (i, "/".join(k)) for i, k in groups) + "]"


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


@contextlib.contextmanager
def capture_debug(logger_name):
    """Collect a logger's DEBUG messages, restoring its level afterwards."""
    lg = logging.getLogger(logger_name)
    handler, old = _Capture(), lg.level
    lg.addHandler(handler)
    lg.setLevel(logging.DEBUG)
    try:
        yield handler.lines
    finally:
        lg.removeHandler(handler)
        lg.setLevel(old)


def capture(retriever=None):
    """One record per query, taken from the dashboard's own answer path."""
    import dashboard.app as app
    from chatbot.intent_guard import THRESHOLD, is_multi_intent, matched_intent_groups
    from chatbot.responder import decision_path
    from chatbot.session_features import SessionTracker

    retriever = retriever if retriever is not None else app._get_retriever()
    records = []
    for n, query in enumerate(QUERIES, 1):
        tracker = SessionTracker("GUARD-FIX-%d" % n, datetime.now(timezone.utc))
        with capture_debug("chatbot.responder") as lines:
            out = app._answer_query(query, tracker, retriever)

        # respond() reports titles only; recover the corpus ids from the same search.
        cands = [{"faq_id": e.index, "title": e.question, "score": s,
                  "category": e.category, "institution": e.institution}
                 for e, s in retriever.search(query, top_k=3)]
        if [c["title"] for c in cands] != [c["question"] for c in out["candidates"]]:
            raise RuntimeError("retrieval differs between calls; cannot attribute FAQ ids")

        groups = matched_intent_groups(query)
        fired = is_multi_intent(query)
        _, reason = decision_path(force_abstain=fired, has_hits=bool(cands),
                                  confidence=out["confidence"],
                                  min_confidence=out["gate"])
        answered = not out["abstained"]
        cited = cands[0] if answered else None
        records.append({
            "n": n, "query": query,
            "top1_score": out["confidence"], "min_confidence": out["gate"],
            "retrieval_backend": out["retrieval_backend"],
            "groups": groups, "n_groups": len(groups), "threshold": THRESHOLD,
            "guard_fired": fired,
            "decision": "answer" if answered else "abstain", "reason": reason,
            "escalate": out["escalate"], "priority": out["priority"],
            "faq_id": cited["faq_id"] if cited else None,
            "title": cited["title"] if cited else None,
            "category": cited["category"] if cited else None,
            "institution": cited["institution"] if cited else None,
            "candidates": cands,
            "debug_line": lines[-1] if lines else "",
        })
    return records


def _guard_cell(r):
    return "FIRED %d>%d" % (r["n_groups"], r["threshold"]) if r["guard_fired"] \
        else "no %d<=%d" % (r["n_groups"], r["threshold"])


def render(records, backend, gate):
    """The report text for ``records`` (pure; no retrieval)."""
    from chatbot.intent_guard import INTENT_GROUPS, THRESHOLD

    rows = [(str(r["n"]), "%.4f" % r["top1_score"], "%.2f" % r["min_confidence"],
             _groups_repr(r["groups"]), _guard_cell(r), r["decision"],
             "#%d" % r["faq_id"] if r["faq_id"] is not None else "-",
             "%s/%s" % (r["escalate"], r["priority"])) for r in records]
    head = ("#", "top1", "gate", "intent_groups", "guard", "decision", "faq", "escalate/priority")
    widths = [max(len(head[i]), *(len(row[i]) for row in rows)) for i in range(len(head))]

    def line(cells):
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths)).rstrip()

    out = [
        "guard_fix_verification - behaviour of %d test queries after guard_institution_fix"
        % len(records),
        "CAPTURE ONLY: no fixes, and no judgement of whether an outcome is right.",
        "Path: dashboard/app.py::_answer_query, fresh tracker per query (a first chat turn).",
        "Reproduce: python scripts/guard_fix_verification.py",
        "",
        "retrieval_backend=%s  min_confidence=%.2f" % (backend, gate),
        "guard: multi-intent when MORE than %d of these topical groups match "
        "(an institution name is not a group):" % THRESHOLD,
    ]
    out += ["  g%d  %s" % (i, ", ".join(g)) for i, g in enumerate(INTENT_GROUPS)]
    out += ["", "SUMMARY", line(head), line(["-" * w for w in widths])]
    out += [line(row) for row in rows]
    out += ["", "DETAIL"]
    for r in records:
        out += [
            "",
            "[%d] %s" % (r["n"], r["query"]),
            "    top1_score        : %s" % r["top1_score"],
            "    min_confidence    : %s" % r["min_confidence"],
            "    intent_groups     : %s  (%d matched; the guard fires above %d)"
            % (_groups_repr(r["groups"]), r["n_groups"], r["threshold"]),
            "    guard_verdict     : %s" % ("FIRED" if r["guard_fired"] else "not fired"),
            "    final_decision    : %s  (reason: %s)" % (r["decision"], r["reason"]),
            "    escalate/priority : %s / %s" % (r["escalate"], r["priority"]),
            "    retrieval_backend : %s" % r["retrieval_backend"],
        ]
        if r["faq_id"] is not None:
            out.append("    cited_faq         : #%d [%s | %s] %s" % (
                r["faq_id"], r["category"], r["institution"], r["title"]))
        else:
            out.append("    cited_faq         : none (abstained)")
        out.append("    retrieval top-3   :")
        for rank, c in enumerate(r["candidates"], 1):
            out.append("      %d. %.4f #%d [%s | %s] %s" % (
                rank, c["score"], c["faq_id"], c["category"], c["institution"], c["title"]))
        out.append("    log               : %s" % r["debug_line"])
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description="guard_fix_verification (capture only)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    import dashboard.app as app
    retriever = app._get_retriever()
    backend = app._backend_of(retriever)
    text = render(capture(retriever), backend, app._gate_for(backend))

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
