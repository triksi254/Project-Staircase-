# Counsellor Assessment Forms — Email Extraction & PII Removal

Automated pipeline that extracts structured, PII-free assessment data from
Jotform-generated `.eml` files under `Forms/`, categorises it, and exports the
results to CSV and JSON.

This package depends only on the Python standard library — no third-party
installs are required.

## Directory layout

```
<repo>/Scrapper/CounsellorForms/          # inside the git repository
├── eml_extractor/                       # extraction package (stdlib-only)
│   ├── parser.py                        # .eml parsing + HTML/plain field extraction
│   ├── categorizer.py                   # Question->standard-category mapping + record build
│   ├── pii.py                           # PII regex patterns + redaction helpers
│   ├── pipeline.py                      # Orchestration (process, validate, export)
│   └── models.py                        # AssessmentRecord dataclass
├── extract_assessment_forms.py          # CLI entry point
├── output/                              # Cleaned CSV + JSON exports (committed corpus)
└── PII_Removal_Checklist.txt            # Manual verification checklist

<repo>/../CounsellorForms/Forms/         # OUTSIDE the repo: raw .eml source files
```

> ⚠️ **Data location.** The raw `.eml` files contain PII and are intentionally
> kept *outside* the git repository (typically at `../CounsellorForms/Forms/`
> relative to the repo root). `extract_assessment_forms.py` auto-locates the
> `Forms/` directory beside the repo; if yours lives elsewhere, pass `--forms DIR`.

## Usage

```bash
python extract_assessment_forms.py
```

Options:

- `--forms DIR` root directory containing `.eml` files (default: auto-located — see *Data location* above)
- `--out DIR` output directory (default: `output/`)
- `--keep-names` keep counsellor names (NOT recommended; PII)

Output files are written as:

- `output/assessment_forms_cleaned_<timestamp>.csv`
- `output/assessment_forms_cleaned_<timestamp>.json`

## What is extracted

| Field             | Source               | PII handling                      |
| ----------------- | -------------------- | --------------------------------- |
| CRM ID            | Email subject / form | Preserved (system ID)             |
| Rating            | Subject / form       | Cold \| Good \| Excellent         |
| Counsellor ID     | Subject              | [NAME_REDACTED] / [COUNSELLOR_ID] |
| Qualifications    | Form                 | Institution names removed         |
| English Test      | Form                 | Kept (no PII)                     |
| Study Destination | Form                 | Country level, kept               |
| Course            | Form                 | Kept                              |
| Intake            | Form                 | Kept (month + year)               |
| Funding Method    | Form                 | Category kept                     |
| Fund Amount       | Form                 | [AMOUNT_REDACTED]                 |
| Assessment Notes  | Form                 | PII redacted                      |

A final automated PII scan is run over the export; any record still
containing emails / phone numbers / passport ranges is flagged.
