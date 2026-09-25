# University Web Scraper

A production-ready web scraping system for collecting institutional data (courses, accommodation, qualifications) from university websites. Features intelligent JavaScript detection, data validation, FAQ integration, and multi-format export.

## 🚀 Quick Start

```bash
# 1. Setup
python -m venv venv
venv\Scripts\activate            # Windows
pip install -r requirements.txt
playwright install chromium      # For JavaScript rendering

# 2. Environment
copy .env.example .env           # Windows

# 3. Scrape a single institution
python main.py --institution aston_university --data-type courses

# 4. Batch scraping
python main.py --batch batch_config_example.json --export json csv
```

## 🏗️ Architecture

```
URL → Smart Router → HTML/JS Scraper → Data Processor → FAQ Integrator → Storage
                          ↓
                    Detection of JS requirement
```

| Component       | Technology        | Purpose                            |
| --------------- | ----------------- | ---------------------------------- |
| HTML Scraping   | BeautifulSoup4    | Parse static HTML content          |
| JS Rendering    | Playwright        | Execute JavaScript and capture DOM |
| Smart Routing   | URL Probing       | Detect if page needs JS rendering  |
| Data Validation | Pydantic          | Type validation and normalization  |
| Logging         | loguru            | Structured application logging     |
| Storage         | SQLite/PostgreSQL | Database storage                   |
| Export          | JSON/CSV          | Multi-format data export           |
| FAQ Integration | Text Similarity   | Cross-reference with FAQ corpus    |

## ⚙️ Configuration

### institutions.yaml

```yaml
institutions:
  aston_university:
    name: "Aston University"
    country: "UK"
    base_url: "https://www.aston.ac.uk"
    data_sources:
      courses:
        url: "https://www.aston.ac.uk/study/find-courses"
        type: "html" # or "javascript"
        paginated: true
        pagination_param: "page"
        max_pages: 50
        delay: 1.5
```

### selectors.yaml

```yaml
aston_university:
  courses:
    container: ".course-item, [data-course-item]"
    fields:
      title: "h2.course-title"
      url: "a.course-link::attr(href)"
      description: ".course-description"
      level: ".qualification-level"
      fees: ".fees-amount::text"
```

**Pseudo-selectors:**

- `::text` — extract text content
- `::attr(name)` — extract HTML attribute
- `::html` — extract inner HTML
- `::all` — extract all matches (list)

## 🎯 Usage

### Single institution

```bash
python main.py --institution aston_university --data-type courses --export json csv
python main.py --institution oxford_university --data-type postgraduate --force-js
python main.py --institution cambridge_university --data-type accommodation --force-html
```

### Batch scraping

```bash
python main.py --batch batch_config_example.json --export json csv database
```

### Distributed workers

```bash
python main.py --batch batch_config.json --worker-id 1
python main.py --batch batch_config.json --worker-id 2
```

### Programmatic usage

```python
from scrapers.smart_scraper import SmartScraper
from processors.data_processor import DataProcessor
from storage.storage import StorageManager
from faq.faq_integrator import FAQIntegrator

scraper = SmartScraper("aston_university", "courses")
raw_data = scraper.scrape()

processor = DataProcessor()
processed_data = processor.process_courses(raw_data)

integrator = FAQIntegrator()
enriched_data = integrator.enrich_records(processed_data, "courses")

storage = StorageManager()
storage.save_results("aston_university", "courses", enriched_data, scraper.metadata, formats=["json", "csv", "database"])
```

## 📁 Project Structure

```
Scrapper/
├── main.py                      # Scraper entry point
├── requirements.txt             # Core dependencies (+ requirements-ml.txt, requirements-app.txt)
│
├── config/  models/  scrapers/  processors/  storage/  faq/   # scraper pipeline (Ch. 3)
│   └── models/                  # Pydantic schemas (package) + classifier artifacts
│
├── CounsellorForms/             # EML assessment-form extraction (see its README)
│
├── leads/                       # Lead scoring (rule + ML hybrid)
│   ├── features.py              # Counsellor-form feature extraction (Cold/Warm/Hot label)
│   ├── rubric.py                # Explainable rule-based scorer
│   ├── personas.py              # Synthetic session generators (v1 hand-set, v2 label-model)
│   ├── train_ml.py              # RF / LogReg, CV, generalization (--group-by-lead)
│   ├── eval_augmentation.py     # real-only / +v1 / +v2 / +distill (--group-split, --render-only)
│   ├── alpha_calibration.py     # train-only, lead-grouped alpha sweep
│   ├── hybrid.py                # alpha*rule + (1-alpha)*ml, live model loading + preprocessing
│   ├── live_config.py           # builds artifacts/config.json (model, alpha, imputation table)
│   └── RESULTS.md / RESULTS_v2.md / RESULTS_v2_grouped.md
│
├── chatbot/                     # Retrieval chatbot
│   ├── retriever.py             # TF-IDF + keyword bonus (KEYWORD_BONUS)
│   ├── embeddings.py            # Sentence-BERT retriever (optional, requirements-ml.txt)
│   ├── classifier.py            # question-type classifier (TF-IDF + LogReg)
│   ├── responder.py             # grounded answer / abstain / escalate (gate constants live here)
│   ├── session_features.py      # live chat -> rubric evidence + engagement features
│   └── demo.py                  # CLI demo
│
├── evaluation/                  # H1 retrieval evaluation
│   ├── retrieval_eval.py  compare_retrievers.py  adaptation_loop.py
│   └── gold_queries.json (A, 39) / gold_queries_b.json (B, 40)   # author-written, NOT blind
│
├── dashboard/app.py             # Streamlit counsellor test platform (chat + live scoring)
│
├── artifacts/                   # metrics/config JSON tracked; *.pkl gitignored
│   ├── config.json              # live-scoring config (model_file, alpha, imputation) - python -m leads.live_config
│   ├── *_corrected.json         # regenerated after fixes; the tagged originals are kept as evidence
│   └── eval_augmentation_grouped*.json, alpha_calibration.json, classifier_cv.json
│
├── data/faq_corpus.json         # 98 FAQ entries (hand-curated, grounded in scraped pages)
├── scratch/                     # one-off exploration scripts (gitignored; corpus assembly lives here)
└── tests/                       # 344 tests (SBERT tests skip without requirements-ml.txt)
```

## 🧠 Lead Scoring Pipeline

```bash
python -m leads.features                      # feature matrix + data-quality warnings
python -m leads.rubric                        # rule-based Cold/Warm/Hot scores
python -m leads.personas --n 1000 --seed 42   # synthetic sessions (v1); --generator v2 for v2
python -m leads.train_ml --cv 5 --cv-repeat 2 [--group-by-lead]
python -m leads.eval_augmentation --seeds 1,7,42 [--group-split]
python -m leads.alpha_calibration             # train-only alpha sweep -> artifacts/alpha_calibration.json
python -m leads.live_config [--check]         # artifacts/config.json for the dashboard
streamlit run dashboard/app.py                # needs requirements-app.txt (+ requirements-ml.txt for SBERT)
```

Headline (full detail: [leads/RESULTS.md](leads/RESULTS.md),
[leads/RESULTS_v2.md](leads/RESULTS_v2.md), [leads/RESULTS_v2_grouped.md](leads/RESULTS_v2_grouped.md)):

- The combined model's **0.90 macro F1 is a confound**: it rides on engagement
  features that exist only in synthetic data. On real leads the defensible figure
  is macro F1 **~0.62-0.64**, Hot F1 ~0.3-0.4 (about 12 Hot rows per holdout).
- **v1 synthetic augmentation hurts** in every configuration. **v2** appeared to help
  in the full feature set (+0.107 mean under the row-level split), but that gain
  does not survive a lead-grouped split (+0.048, seed range [-0.012, +0.092]) and is
  absent in the fair no-engagement set (<= 0 at all three seeds when grouped).
- The blend weight **alpha = 0.5 is an uncalibrated default**. The train-only,
  lead-grouped calibration (`artifacts/alpha_calibration.json`) finds alpha = 0.1
  best (macro F1 0.623) versus 0.471 at 0.5 and 0.166 for the rubric alone: on
  counsellor labels the form rubric adds nothing (it agrees with the counsellor
  ~19% of the time). This does not transfer to the dashboard's chat rule score.

### Running the dashboard

```bash
streamlit run dashboard/app.py
```

The first start loads the Sentence-BERT model (a few seconds; a `Loading weights`
progress bar is normal). The page header states which retriever is active
(`Retrieval: sbert · abstains below 0.60`); if SBERT cannot start you get a warning
and TF-IDF at a lower gate. The terminal should be nearly silent. Two messages are
benign and **not from the app**: the `Loading weights` progress bar, and on Windows a
`ConnectionResetError: [WinError 10054]` from asyncio when a browser tab closes
abruptly. (Importing `transformers` used to make Streamlit's file watcher log ~400
tracebacks about a missing `torchvision`; `dashboard/app.py` filters exactly those
records, and the model now loads from the local HuggingFace cache first, so there is
no "unauthenticated requests" warning.)

## 🔎 Known limitations (read before citing numbers)

- **Repeated leads.** 39% of the 861 labelled rows (336) belong to a CRM id that
  occurs more than once, almost always with the same label. A row-level split puts a
  same-lead sibling in train for 38-42% of holdout rows. `--group-by-lead` /
  `--group-split` remove this; the default splits are unchanged so tagged results
  stay reproducible. `python -m leads.features` prints this warning.
- **Sparse features.** Only 127 distinct profile vectors among 861 rows; 61% of rows
  sit in a vector with conflicting labels. `assessment_notes` is empty in all 964
  records (so `note_word_count` is constant). `passport_status = valid` is a
  *default* (94% of rows, all Hot), not an observation.
- **Live scoring is provisional.** The dashboard's rule score covers only what a chat
  can observe (course / intake / English topics); unobservable fields are imputed from
  the training table in `artifacts/config.json`, never assumed negative. Its ML score
  is `lead_model_rf_full.pkl`, an *engagement-persona* model (~71% of importance on
  synthetic-only engagement features), not a validated conversion model. The `*.pkl`
  files are gitignored: on a clean clone the dashboard is rule-only and the
  classifier falls back to "General Enquiries".
- **Retrieval evaluation.** The gold sets are author-written against an author-built
  corpus (not blind); 65-84% of answerable queries contain a keyword of their own gold
  entry. SBERT-vs-TF-IDF differences are not statistically significant on either set.
  The TF-IDF abstention metrics depend on `KEYWORD_BONUS`: the tagged artifacts used
  0.05, the code later moved to 0.10 (see `artifacts/compare_gold_*_corrected.json`).
  The adaptation loop cannot discriminate targeted from full restoration with 8
  withheld entries over 9 categories (`artifacts/adaptation_results_corrected.json`).
- **Data governance.** `CounsellorForms/output/*.json` retains numeric CRM ids
  (`PII_Removal_Checklist.txt` says "preserved"), whereas the research proposal
  promised to remove them - a decision for the author / ethics sign-off. These
  exports were tracked in git from `cfdac78` onward and are now untracked
  (`CounsellorForms/output/.gitignore`), but remain in that history and in any
  existing clone or fork; untracking does not retroactively remove them.

## Deviations from the proposal

| Proposal | As built |
|---|---|
| SBERT + FAISS, live web-search (Tavily) fallback, weekly refresh | SBERT with exact dot product (no FAISS); no web fallback; no scheduler |
| DistilBERT classifier arm; XGBoost | TF-IDF + LogReg only; RF + LogReg |
| "Only synthetic" lead-scoring (RQ2), n~50-100 real forms | 861 real counsellor rows + synthetic; no synthetic-only run |
| H1 on cosine vs gold *answers* > 0.80 | Query-to-document raw cosine (falsified) + rank metrics (post hoc) |
| H2 on synthetic personas, F1 > 0.80 | Real counsellor labels (not met: ~0.62-0.64) |
| H3 (score rises with intent) tested with ANOVA | Not a statistical test; scripted demo only |

## ✨ Key Features

1. **Smart JavaScript Detection** — analyzes HTML for React/Vue/Angular indicators, API placeholders, sparse content; falls back to HTML scraper when possible. Override with `--force-js` / `--force-html`.
2. **Data Validation** — Pydantic models with automatic normalization of qualification levels and study modes.
3. **Pagination** — multi-page support, configurable limits, smart next-page detection.
4. **FAQ Integration** — keyword + text-similarity matching against `data/faq_corpus.json`.
5. **Multi-Format Export** — JSON (full + metadata), CSV (flattened), SQLite/PostgreSQL.
6. **Error Handling** — retries with exponential backoff, 429 detection, screenshots on JS errors.
7. **Monitoring & Logging** — structured loguru logging to console and file.

## 🐛 Troubleshooting

**"No items found with selector"** → inspect page, update `selectors.yaml`, test with `--force-html`.

**JS content not loading** → use `--force-js`, set `type: "javascript"`, increase `WAIT_SELECTOR_TIMEOUT`.

**Playwright crashes** → reinstall: `pip uninstall playwright && pip install playwright && playwright install chromium`.

**Rate limited (429)** → increase delay: `--delay 3.0`.

## 📊 Sample Output

```json
{
  "metadata": {
    "institution": "aston_university",
    "data_type": "courses",
    "scraped_at": "2024-01-15T10:30:00",
    "total_items": 127,
    "total_errors": 3
  },
  "data": [
    {
      "title": "BSc Computer Science",
      "url": "https://www.aston.ac.uk/study/courses/bsc-computer-science",
      "description": "A comprehensive degree in computer science...",
      "level": "bachelor",
      "duration": "3 years",
      "study_type": "full_time",
      "fees": "£9250",
      "related_faqs": [
        {
          "question": "How long does the course take?",
          "answer": "Most undergraduate courses take 3-4 years...",
          "relevance_score": 0.92
        }
      ]
    }
  ]
}
```

## 🐛 Best Practices

- Respect `robots.txt` and use reasonable delays
- Set a proper User-Agent identifying your scraper
- Prefer the HTML scraper when possible (faster)
- Deduplicate entries before storage
- Monitor `logs/scraper.log` regularly

## 📝 License

MIT License

## 🗺️ Roadmap

- [ ] Async scraping support
- [ ] Distributed scraping with Celery
- [ ] Web dashboard for monitoring
- [ ] Advanced search indexing
- [ ] Real-time data updates
- [ ] Machine learning for field extraction
- [ ] GraphQL API for data access
