"""scripts/guard_fix_verification.py: capture only.

The diagnostic records how eight queries behave after ``guard_institution_fix``
without judging them. These tests therefore pin the *shape* of the report and its
internal consistency (guard verdict vs groups, decision vs gate, cited FAQ vs
retrieval, quoted log line vs record), never whether a query's outcome is right.
"""
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "guard_fix_verification.py"

REQUESTED = [
    "Do I need IELTS to study at Aston if I did KCSE English?",
    "What are the entry requirements for University of Birmingham?",
    "I need to know about IELTS, scholarships, and accommodation",
    "Which university should I study at in the UK?",
    "I have a Kenyan passport. What do I need for a masters in the UK?",
    "I have a Kenyan passport, IELTS 6.5, and want to study Data Science",
    "How much does accommodation cost at Aston University?",
    "What IELTS score do I need for a masters at RGU?",
]
FIELDS = ("top1_score", "min_confidence", "intent_groups", "guard_verdict",
          "final_decision", "retrieval_backend", "cited_faq")


def _load():
    spec = importlib.util.spec_from_file_location("guard_fix_verification", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake(n, answered):
    return {
        "n": n, "query": "query number %d" % n, "top1_score": 0.7123 if answered else 0.4,
        "min_confidence": 0.6, "retrieval_backend": "sbert",
        "groups": [(1, ["ielts"]), (3, ["masters"])], "n_groups": 2, "threshold": 2,
        "guard_fired": False, "decision": "answer" if answered else "abstain",
        "reason": "top1 0.7123 >= min_confidence 0.6; no guard" if answered
        else "low_confidence (top1 0.4 < min_confidence 0.6)",
        "escalate": not answered, "priority": "none" if answered else "normal",
        "faq_id": 49 if answered else None,
        "title": "Does Aston accept KCSE English instead of IELTS?" if answered else None,
        "category": "English Language" if answered else None,
        "institution": "Aston" if answered else None,
        "candidates": [{"faq_id": 49, "title": "Does Aston accept KCSE English instead "
                        "of IELTS?", "score": 0.7123, "category": "English Language",
                        "institution": "Aston"}],
        "debug_line": "respond: query='q' top1_score=0.7123",
    }


def test_the_eight_queries_are_the_requested_ones_in_order():
    assert list(_load().QUERIES) == REQUESTED


def test_render_lists_every_requested_field_for_every_query():
    m = _load()
    recs = [_fake(1, True), _fake(2, False)]
    text = m.render(recs, backend="sbert", gate=0.6)
    for field in FIELDS:
        assert text.count(field) >= len(recs), field
    assert "#49" in text
    assert "Does Aston accept KCSE English instead of IELTS?" in text
    assert "none (abstained)" in text                       # the abstained record


def test_report_is_internally_consistent_on_the_real_retriever(tmp_path):
    import dashboard.app as app
    m = _load()
    recs = m.capture()
    assert [r["query"] for r in recs] == REQUESTED
    for r in recs:
        backend = r["retrieval_backend"]
        assert r["min_confidence"] == app._gate_for(backend)
        assert r["guard_fired"] == (len(r["groups"]) > r["threshold"])
        if r["guard_fired"] or r["top1_score"] < r["min_confidence"]:
            assert r["decision"] == "abstain", r["query"]
        if r["decision"] == "answer":
            assert isinstance(r["faq_id"], int) and r["title"], r["query"]
            assert r["title"] == r["candidates"][0]["title"]
        else:
            assert r["faq_id"] is None and r["title"] is None, r["query"]
        # the respond() log line quoted in the report agrees with the record
        assert "top1_score=%r" % r["top1_score"] in r["debug_line"]
        assert "decision=%s" % r["decision"] in r["debug_line"]
        assert m._groups_repr(r["groups"]) in r["debug_line"]

    out = tmp_path / "guard_fix_verification.txt"
    assert m.main(["--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    for q in REQUESTED:
        assert q in text
