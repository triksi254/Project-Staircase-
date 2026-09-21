"""diagnose_v3 - three diagnostics on the recent counsellor-platform test session.

DIAGNOSIS ONLY: nothing here changes behaviour. The queries and category
sequences come from three exported chats (LEAD-20260921-0001..0003).

  1. ESCALATION   Why did answers shown at confidence 0.81 (gate 0.60) escalate?
  2. RULE SCORE   Why do LEAD-0001 and LEAD-0002 both end at exactly 0.500, and
                  where is the ceiling?
  3. CATEGORIES   Where did the three "wrong" categories actually come from?

Every respond() call replicates ``dashboard/app.py::_answer_query`` (SBERT
retriever, its backend gate, ``force_abstain = is_multi_intent(query)``).

Usage:
    python scripts/diagnose_v3.py [--out artifacts/diagnose_v3.txt]
"""
from __future__ import annotations

import argparse
import contextlib
import inspect
import itertools
import logging
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / "artifacts" / "diagnose_v3.txt"

# --- item 1: the two queries that escalated at displayed confidence 0.81 -------
ITEM1 = [
    "Do I need IELTS to study at Aston if I did KCSE English?",
    "What IELTS score do I need for a masters at RGU?",
]

# --- item 2: the categories recorded on the tracker in the three exported chats
SESSIONS = {
    "LEAD-20260921-0001": ["Scholarships", "English Language", "Application Process"],
    "LEAD-20260921-0002": ["Accommodation", "English Language", "Application Process"],
    "LEAD-20260921-0003": ["English Language", "Scholarships", "Visa & Immigration"],
}
#: rubric components a chat can observe, and the FAQ category that sets each one
TOPICS = (("Entry Requirements", "has_course"),
          ("Application Process", "has_intake"),
          ("English Language", "english_test"))

# --- item 3: (tag, query, category you expected, category the session assigned) -
ITEM3 = [
    ("A", "How much is tuition at BCU for a postgraduate course?",
     "Fees & Funding", "Scholarships"),
    ("B", "Is it safe to study in Birmingham as an international student?",
     "General Enquiries", "Application Process"),
    ("C", "Can I transfer from a Kenyan university to a UK university mid-degree?",
     "Application Process", "Visa & Immigration"),
]


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


def dashboard_call(query, retriever, gate):
    """respond() exactly as dashboard/app.py::_answer_query calls it."""
    from chatbot.intent_guard import is_multi_intent
    from chatbot.responder import EscalationPolicy, respond
    return respond(query, retriever, rule_score=0.0, ml_score=None,
                   policy=EscalationPolicy(alpha=1.0, min_confidence=gate),
                   top_k=3, force_abstain=is_multi_intent(query))


# ------------------------------------------------------------------ item 2 helpers
def english_credit(band):
    """The rubric's english_test contribution for a chat-style row."""
    from leads.rubric import _english_contrib
    return _english_contrib({"has_english_test": 1, "english_band": band})


def _tracker(cats, name="x"):
    from chatbot.session_features import SessionTracker
    t0 = datetime(2026, 9, 21, 22, 0, 0)
    tr = SessionTracker(name, t0)
    for i, c in enumerate(cats):
        tr.add_turn("q%d" % i, {"category": c, "confidence": 0.8},
                    t0 + timedelta(seconds=30 * i))
    return tr


def rule_lattice():
    """Every reachable live rule score: one row per subset of the 3 topics."""
    from chatbot.session_features import live_rule_breakdown
    names = [t[0] for t in TOPICS]
    rows = []
    for r in range(len(names) + 1):
        for combo in itertools.combinations(names, r):
            bd = live_rule_breakdown(_tracker(combo, "lattice").rubric_evidence())
            rows.append({"topics": tuple(combo), "raw": bd["raw"],
                         "mass": bd["mass"], "score": bd["score"]})
    return rows


def _line_of(fn, needle):
    src, start = inspect.getsourcelines(fn)
    for off, line in enumerate(src):
        if needle in line:
            return "%s:%d" % (Path(inspect.getsourcefile(fn)).relative_to(ROOT).as_posix(),
                              start + off)
    return Path(inspect.getsourcefile(fn)).relative_to(ROOT).as_posix()


# ------------------------------------------------------------------------ report
def part1(emit, retriever, gate):
    from chatbot.intent_guard import THRESHOLD, matched_intent_groups
    emit("=" * 96)
    emit("1. ESCALATION - why did confident answers escalate?")
    emit("=" * 96)
    emit("retriever=%s gate=%.2f. Each call replicates dashboard/app.py::_answer_query:" % (
        retriever.backend, gate))
    emit("  respond(query, retriever, rule_score=0.0, ml_score=None,")
    emit("          policy=EscalationPolicy(alpha=1.0, min_confidence=gate), top_k=3,")
    emit("          force_abstain=is_multi_intent(query))")
    emit("")
    for q in ITEM1:
        with capture_debug("chatbot.responder") as lines:
            out = dashboard_call(q, retriever, gate)
        for ln in lines:
            emit("DEBUG chatbot.responder: " + ln)
        groups = matched_intent_groups(q)
        top1 = out["confidence"]
        idx = {e.question: e.index for e in retriever.entries}
        emit("  retrieval top-3:")
        for c in out["candidates"]:
            emit("     %.4f #%s [%s] %s" % (c["score"], idx.get(c["question"], "?"),
                                          c["category"], c["question"][:70]))
        emit("  confidence gate      : top1 %.4f %s %.2f -> %s" % (
            top1, ">=" if top1 >= gate else "<", gate,
            "did NOT fire" if top1 >= gate else "FIRED"))
        emit("  multi-intent guard   : %s = %d groups %s %d -> %s" % (
            " + ".join("g%d:%s" % (i, "/".join(k)) for i, k in groups), len(groups),
            ">" if len(groups) > THRESHOLD else "<=", THRESHOLD,
            "FIRED" if len(groups) > THRESHOLD else "did not fire"))
        emit("  institution scope    : not invoked (no such check exists at HEAD)")
        emit("  final                : %s -> escalate=%s priority=%s" % (
            "abstain" if out["abstained"] else "answer", out["escalate"],
            out["priority"]))
        emit("")
    emit("Scope-check history (git): added in b940388 (2026-09-19); removed in a6832c6")
    emit("(2026-09-20, the 'empty lead state' commit). At HEAD nothing filters or gates by")
    emit("institution. Institution names used to count as intent group g5 of the")
    emit("multi-intent guard, which is what escalated both queries when this was first")
    emit("diagnosed; an institution is now a scope qualifier and no longer counts.")
    emit("")


def part2(emit):
    emit("=" * 96)
    emit("2. RULE SCORE CEILING - raw score before normalisation")
    emit("=" * 96)
    emit("live_rule_breakdown() computes: score = raw / mass  (no clamp anywhere on the path).")
    emit("")
    for lead, cats in SESSIONS.items():
        tr = _tracker(cats, lead)
        with capture_debug("chatbot.session_features") as lines:
            score = tr.rule_score()
        emit("%s categories=%s" % (lead, cats))
        for ln in lines[-1:]:
            emit("   DEBUG chatbot.session_features: " + ln)
        emit("   rule_score = %.4f" % score)
    emit("")
    emit("Every reachable live rule score (subsets of the 3 observable topics):")
    emit("   %-52s %7s %7s %8s" % ("topics seen (-> component)", "raw", "mass", "score"))
    lat = rule_lattice()
    comp = dict(TOPICS)
    for r in lat:
        label = ", ".join("%s(%s)" % (t, comp[t]) for t in r["topics"]) or "(none)"
        emit("   %-52s %7.4f %7.4f %8.4f" % (label[:52], r["raw"], r["mass"], r["score"]))
    distinct = sorted({round(r["score"], 4) for r in lat})
    top = max(lat, key=lambda r: r["score"])
    emit("")
    emit("distinct values: %d  %s" % (len(distinct), distinct))
    emit("ceiling: raw %.4f / mass %.4f = %.4f (never 1.0)" % (top["raw"], top["mass"], top["score"]))
    emit("  why not 1.0: english_test earns %.3f of its %.3f weight (band unparsed);" % (
        english_credit(0.0), english_credit(6.5)))
    emit("  with a parsed band >= 6.0 it would earn %.3f, but a chat never supplies a band." % english_credit(6.5))
    emit("  half credit set at        %s" % _line_of(__import__("leads.rubric", fromlist=["x"])._english_contrib, "band == 0.0"))
    emit("  band fixed at 0.0 by      %s" % _line_of(__import__("chatbot.session_features", fromlist=["x"]).empty_evidence, '"english_band"'))
    emit("why 0.5000 twice: Application Process+English and Entry Requirements+English both")
    emit("  have raw 0.09 (0.05 + 0.04) over mass 0.18. The score is a 6-value lattice, so")
    emit("  different chats collide. course+intake without English gives 0.5556.")
    emit("clamps on the path: leads.hybrid._clip01 in hybrid_score(); it does not bind")
    emit("  (rule <= 0.7778, hybrid in [0,1]). Nothing else clamps.")
    emit("")


def part3(emit, retriever, gate):
    from chatbot.classifier import load_training_data, predict_distribution
    emit("=" * 96)
    emit("3. CATEGORY DRIFT - where did each category come from?")
    emit("=" * 96)
    emit("respond() takes a turn's category from the RETRIEVED FAQ when it answers, and from")
    emit("the CLASSIFIER only when it abstains (chatbot/responder.py).")
    emit("")
    _, labels = load_training_data()
    counts = Counter(labels)
    idx = {e.question: e.index for e in retriever.entries}
    flat = []
    for tag, q, expected, assigned in ITEM3:
        out = dashboard_call(q, retriever, gate)
        dist = predict_distribution(q)
        rank = {c: i + 1 for i, (c, _) in enumerate(dist)}
        r_exp = rank.get(expected)
        retr = out["candidates"]
        retr_cats = [c["category"] for c in retr]
        retr_rank = retr_cats.index(expected) + 1 if expected in retr_cats else None
        from_retrieval = not out["abstained"]
        emit("[%s] %s" % (tag, q))
        emit("    expected category      : %s" % expected)
        emit("    assigned in session    : %s" % assigned)
        emit("    respond() path         : %s (top1 %.4f, gate %.2f)" % (
            "answer" if from_retrieval else "abstain", out["confidence"], gate))
        if from_retrieval:
            emit("    category source: retrieved FAQ -> #%s [%s] %s" % (
                idx.get(retr[0]["question"], "?"), retr[0]["category"], retr[0]["question"][:60]))
            emit("                     (the classifier was NOT consulted for this turn)")
        else:
            emit("    category source: classifier fallback -> %s" % out["category"])
        emit("    retrieval top-3        : " + " | ".join(
            "%.4f [%s]" % (c["score"], c["category"]) for c in retr))
        emit("    classifier distribution (full, best first):")
        emit("       " + " | ".join("%s %.3f" % (c, p) for c, p in dist))
        emit("    classifier rank of expected: %s of %d  (top-3: %s, top-1: %s)" % (
            r_exp, len(dist), "yes" if r_exp and r_exp <= 3 else "NO",
            "yes" if r_exp == 1 else "no"))
        flat.append((dist[0][1], dist[0][1] - dist[1][1]))
        if from_retrieval:
            emit("    verdict: NOT classifier drift - the category is the retrieved FAQ's.")
            if retr_rank == 1:
                emit("             the retrieved FAQ's category already equals the expected one.")
            elif retr_rank:
                emit("             expected category is at retrieval rank %d (gap to top-1 %.4f):" % (
                    retr_rank, retr[0]["score"] - retr[retr_rank - 1]["score"]))
                emit("             a retrieval-ranking issue, not a training-data gap.")
            else:
                emit("             expected category is absent from the retrieval top-3:")
                emit("             corpus coverage / gate issue, not a classifier issue.")
            emit("             (had the classifier been used it would say %s; expected is rank %s)" % (
                dist[0][0], r_exp))
        elif r_exp == 1:
            emit("    verdict: classifier top-1 is the expected category.")
        elif r_exp is not None and r_exp <= 3:
            emit("    verdict: expected is in the classifier top-3 but not top-1 -> threshold issue.")
        else:
            emit("    verdict: expected is ABSENT from the classifier top-3 (rank %s of %d) ->" % (
                r_exp, len(dist)))
            emit("             training-data coverage issue (%d training docs for '%s')." % (
                counts.get(expected, 0), expected))
        emit("")
    emit("Classifier flatness: top-1 probability %s; margin top1-top2 %s (uniform over 9 = 0.111)." % (
        "/".join("%.3f" % a for a, _ in flat), "/".join("%.3f" % b for _, b in flat)))
    emit("Training docs per category: " + ", ".join(
        "%s=%d" % (c, n) for c, n in sorted(counts.items(), key=lambda t: t[1])))
    emit("")


def main(argv=None):
    ap = argparse.ArgumentParser(description="diagnose_v3 (diagnosis only)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    import dashboard.app as app
    retriever = app._get_retriever()
    gate = app._gate_for(app._backend_of(retriever))

    lines = []

    def emit(text=""):
        lines.append(text)
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode("ascii", "replace").decode("ascii"))

    part1(emit, retriever, gate)
    part2(emit)
    part3(emit, retriever, gate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    emit("Wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
