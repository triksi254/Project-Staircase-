"""Tests for the chatbot retrieval layer (grounded, abstaining)."""
from chatbot.responder import EscalationPolicy, respond
from chatbot.retriever import FaqEntry, Retriever, load_corpus, tokenize
from leads.hybrid import score_to_label


def _toy() -> Retriever:
    return Retriever([
        FaqEntry(question="Do I need a student visa?",
                 answer="Yes, you need a Student visa with a CAS.",
                 keywords=["visa", "cas"]),
        FaqEntry(question="Are scholarships available?",
                 answer="Yes, merit scholarships exist.",
                 keywords=["scholarship"]),
        FaqEntry(question="Where will I live?",
                 answer="University halls and private rentals.",
                 keywords=["accommodation"]),
    ])


def test_tokenize_strips_stopwords():
    assert "visa" in tokenize("Do I need a Visa?")
    assert "the" not in tokenize("the visa")


def test_search_ranks_visa_first():
    r = _toy()
    hits = r.search("student visa CAS requirements", top_k=2)
    assert hits[0][0].question.startswith("Do I need")
    assert hits[0][1] >= hits[1][1]


def test_search_empty_query_returns_empty():
    assert _toy().search("   ") == []


def test_load_corpus_missing_warns(tmp_path, caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="chatbot.retriever"):
        assert load_corpus(tmp_path / "nope.json") == []
    assert "missing" in caplog.text


def test_real_corpus_grounded_answers():
    entries = load_corpus()
    assert len(entries) == 98
    r = Retriever(entries)
    hits = r.search("Do international students need a visa?", top_k=1)
    assert hits and "visa" in (hits[0][0].question + hits[0][0].answer).lower()


def test_responder_answers_and_cites():
    out = respond("student visa CAS", _toy(), rule_score=0.5, ml_score=0.5)
    assert "visa" in out["answer"].lower()
    assert out["cited_question"] is not None
    assert out["escalate"] is False


def test_responder_abstains_when_unsure():
    out = respond("quantum hamster tax", _toy(), rule_score=0.1, ml_score=0.1,
                  policy=EscalationPolicy(min_confidence=0.9))
    assert out["cited_question"] is None
    assert out["escalate"] is True


def test_hot_lead_escalates_with_priority():
    out = respond("student visa CAS", _toy(), rule_score=1.0, ml_score=1.0)
    assert out["lead_label"] == "Hot"
    assert out["escalate"] is True
    assert out["priority"] == "high"


def test_escalation_matches_label_boundary():
    # score_to_label tertiles: Cold < 1/3, Warm < 2/3, Hot >= 2/3.
    assert score_to_label(2.0 / 3.0) == "Hot"
    assert score_to_label(0.66) == "Warm"
    out = respond("student visa CAS", _toy(), rule_score=2.0 / 3.0,
                  ml_score=2.0 / 3.0)
    assert out["lead_label"] == "Hot"
    assert out["priority"] == "high"
    assert out["escalate"] is True
    warm = respond("student visa CAS", _toy(), rule_score=0.5, ml_score=0.5)
    assert warm["lead_label"] == "Warm"
    assert warm["priority"] != "high"


def test_two_retrievers_agree():
    entries = load_corpus()
    r1, r2 = Retriever(entries), Retriever(entries)
    q = "Do international students need a visa?"
    assert [e.question for e, _ in r1.search(q)] == [e.question for e, _ in r2.search(q)]
    assert [s for _, s in r1.search(q)] == [s for _, s in r2.search(q)]


def test_respond_returns_category_and_institution():
    entries = load_corpus()
    out = respond("Do international students need a visa?", Retriever(entries),
                  rule_score=0.5, ml_score=0.5)
    assert out["category"] == "Visa & Immigration"
    assert out["institution"] in ("General", "USW", "Aston")
