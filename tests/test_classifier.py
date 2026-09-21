"""Tests for the question-type classifier (9 categories, thin data)."""
from chatbot.classifier import (
    CATEGORIES,
    cross_validate,
    load_training_data,
    predict,
    save,
    train,
)


def test_classifier_trains_and_covers_9_categories():
    texts, labels = load_training_data()
    assert len(texts) == 98
    assert sorted(set(labels)) == sorted(CATEGORIES)
    model = train(texts, labels)
    assert model is not None
    save(model, CATEGORIES)


def test_predict_known_category_fees():
    texts, labels = load_training_data()
    model = train(texts, labels)
    assert predict("how much are the tuition fees", model) == "Fees & Funding"


def test_predict_visa():
    texts, labels = load_training_data()
    model = train(texts, labels)
    assert predict("do I need a visa to study in the UK", model) == "Visa & Immigration"


def test_predict_unknown_returns_default():
    texts, labels = load_training_data()
    model = train(texts, labels)
    assert predict("zxqwkj blorpt fnord", model) in CATEGORIES
    assert predict("", model) == "General Enquiries"


def test_cv_macro_f1_reported():
    texts, labels = load_training_data()
    out = cross_validate(texts, labels, k=5)
    print(f"\nCV macro F1: {out['mean_macro_f1']:.4f} folds={out['folds']}")
    assert len(out["folds"]) == 5
    assert 0.0 <= out["mean_macro_f1"] <= 1.0
    # Thin-data expectation: report actual, gate low to avoid brittle CI.
    assert out["mean_macro_f1"] > 0.50


def test_cli_writes_the_cv_summary_artifact(tmp_path):
    import json

    from chatbot.classifier import main

    out = tmp_path / "cv.json"
    assert main(["--out", str(out)]) == 0
    d = json.loads(out.read_text(encoding="utf-8"))
    assert d["k"] == 5 and d["n_documents"] == 98 and d["n_categories"] == 9
    assert len(d["folds"]) == 5
    assert 0.0 < d["mean_macro_f1"] <= 1.0
    assert "NOT a measurement on user queries" in d["note"]
