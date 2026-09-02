# Counsellor Forms — Extraction Pipeline Progress

Status: COMPLETE (core pipeline)

## Completed Steps

- [x] Locate .eml source folders (Assessment Forms _ + Own(Bunch_)) under Forms/
- [x] Build stdlib-only extraction package in `eml_extractor/`
  - [x] `parser.py` - email.message parse + HTML `emailFieldsTable` extraction + plain-text fallback
  - [x] `categorizer.py` - question->standard-category mapping + record build + quality flag
  - [x] `pii.py` - regex patterns for email/phone/passport/IP/amount/address/social/names
  - [x] `pipeline.py` - orchestration + CSV/JSON export + final PII scan
  - [x] `models.py` - AssessmentRecord dataclass
- [x] CLI entry point `extract_assessment_forms.py`
- [x] Validate on sample Kenya & UK Office / Pre-Application variants
- [x] Run full batch (964 .eml files processed)
- [x] Zero PII leaks confirmed on final export

## Data Quality Summary (batch run)

- Total records: 964
- Ratings: Good 426, Cold 374, Excellent 61, (empty/NA 103)
- Quality flags: CLEAN 848, CHECK 116
- PII leaks detected: 0

## PII Hardening (2026-09-02)

- Added contextual counsellor-name redaction: names in the standard
  "Counsellor Assessment Form - <Name> - <Rating>" form are now redacted even
  when the counsellor is NOT in the hard-coded name list (previously an unseen
  name could leak into `source_file` / `counsellor_id`).
- Hardened `categorizer.py` so a bare counsellor identity can never remain in
  the record when names are being redacted.
- Extended `AMOUNT_PATTERN` to catch currency words ("17000 pounds") and bare
  comma-separated amounts ("300,000", "21,000", "16,600 GBP"). Regenerated the
  corpus: 3 records had leaked amounts that are now `[AMOUNT_REDACTED]`; the
  other 961 records are byte-identical to the previous export.
- Added unit tests for the EML extractor (`tests/test_eml_extractor.py`).

## Recommended Follow-ups

- [ ] Manual peer review on ~10% of records (per guide best practice)
- [ ] Extend config-driven PII patterns in `pii.py` as needed
- [ ] Optionally add `.xlsx` export (requires `openpyxl`)
- [ ] Archive original .eml files after validation (retention policy)
