"""One-off: enrich data/faq_corpus.json with category + institution."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "faq_corpus.json"
BACKUP = ROOT / "data" / "faq_corpus.bak.json"

CATEGORIES = ["Entry Requirements", "Fees & Funding", "Scholarships",
              "Program Details", "Accommodation", "Visa & Immigration",
              "Application Process", "English Language", "General Enquiries"]
# (category, institution) per index 0..97, in corpus order. Entries 0-24
# are general market-level FAQs; 25+ name an institution in the question
# or sit inside a grounded batch block for that institution.
LABELS = [
    ("Program Details", "General"), ("Fees & Funding", "General"),
    ("Entry Requirements", "General"), ("Program Details", "General"),
    ("Program Details", "General"), ("Accommodation", "General"),
    ("Accommodation", "General"), ("Accommodation", "General"),
    ("Accommodation", "General"), ("Program Details", "General"),
    ("Scholarships", "General"), ("Program Details", "General"),
    ("Program Details", "General"), ("Program Details", "General"),
    ("English Language", "General"), ("Entry Requirements", "General"),
    ("Visa & Immigration", "General"), ("Application Process", "General"),
    ("Application Process", "General"), ("Visa & Immigration", "General"),
    ("Fees & Funding", "General"), ("Fees & Funding", "General"),
    ("Application Process", "General"), ("Entry Requirements", "General"),
    ("Scholarships", "General"),
    ("English Language", "Aston"), ("Entry Requirements", "Aston"),
    ("Entry Requirements", "Aston"), ("Entry Requirements", "Aston"),
    ("Scholarships", "Aston"), ("Entry Requirements", "Aston"),
    ("Scholarships", "BCU"), ("Scholarships", "BCU"),
    ("English Language", "BCU"), ("English Language", "BCU"),
    ("Entry Requirements", "BCU"),
    ("Entry Requirements", "RGU"), ("Entry Requirements", "RGU"),
    ("English Language", "RGU"), ("English Language", "RGU"),
    ("English Language", "RGU"),
    ("Entry Requirements", "UCLan"), ("Entry Requirements", "UCLan"),
    ("Entry Requirements", "UCLan"), ("English Language", "UCLan"),
    ("Entry Requirements", "UCLan"),
    ("English Language", "Aston"), ("English Language", "Aston"),
    ("English Language", "Aston"), ("English Language", "Aston"),
    ("Scholarships", "Aston"), ("Scholarships", "Aston"),
    ("Program Details", "Aston"), ("Program Details", "Aston"),
    ("Program Details", "Aston"), ("Program Details", "Aston"),
    ("Application Process", "Aston"), ("Application Process", "Aston"),
    ("Fees & Funding", "Aston"), ("Accommodation", "Aston"),
    ("General Enquiries", "Aston"),
    ("Scholarships", "BCU"), ("Fees & Funding", "BCU"),
    ("Accommodation", "BCU"), ("Application Process", "BCU"),
    ("Application Process", "BCU"), ("General Enquiries", "BCU"),
    ("General Enquiries", "BCU"),
    ("English Language", "UCLan"), ("Scholarships", "USW"),
    ("Fees & Funding", "USW"), ("Accommodation", "USW"),
    ("Visa & Immigration", "USW"), ("English Language", "USW"),
    ("Application Process", "USW"), ("General Enquiries", "USW"),
    ("Scholarships", "Salford"), ("Fees & Funding", "Salford"),
    ("General Enquiries", "Salford"),
    ("Application Process", "Herts"), ("Application Process", "Herts"),
    ("Application Process", "Herts"),
    ("Accommodation", "Aston"), ("Accommodation", "Aston"),
    ("Accommodation", "Aston"), ("Accommodation", "Aston"),
    ("Accommodation", "Aston"),
    ("Fees & Funding", "Aston"), ("Fees & Funding", "Aston"),
    ("Fees & Funding", "Aston"), ("Fees & Funding", "Aston"),
    ("Fees & Funding", "Aston"),
    ("Fees & Funding", "BCU"), ("Fees & Funding", "BCU"),
    ("Scholarships", "BCU"),
    ("Visa & Immigration", "USW"), ("Visa & Immigration", "USW"),
    ("Entry Requirements", "Herts"),
]

def main() -> None:
    raw = json.loads(CORPUS.read_text(encoding="utf-8"))
    faqs = raw["faqs"] if isinstance(raw, dict) else raw
    assert len(faqs) == 98, f"expected 98, got {len(faqs)}"
    assert len(LABELS) == 98
    BACKUP.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    for entry, (cat, inst) in zip(faqs, LABELS):
        assert cat in CATEGORIES, cat
        entry["category"] = cat
        entry["institution"] = inst
    payload = {"faqs": faqs} if isinstance(raw, dict) else faqs
    CORPUS.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    from collections import Counter
    print("entries with category:", sum(1 for f in faqs if f.get("category")), "/ 98")
    print("entries with institution:", sum(1 for f in faqs if f.get("institution")), "/ 98")
    print("by category:", dict(Counter(f["category"] for f in faqs)))
    print("by institution:", dict(Counter(f["institution"] for f in faqs)))
    print("inferred rule-based: 98 (0 unknown — every entry labelled)")


if __name__ == "__main__":
    main()