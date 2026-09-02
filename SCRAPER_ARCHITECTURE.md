# Scraper Architecture

This document details the internal architecture of the University Web Scraper.

## Module Hierarchy

```
main.py (Entry Point)
  ├── SmartScraper (Route to appropriate scraper)
  │   ├── HTMLScraper (BeautifulSoup)
  │   │   └── BaseScraper
  │   └── JavaScriptScraper (Playwright)
  │       └── BaseScraper
  ├── DataProcessor (Clean & Validate)
  │   └── Pydantic Models (Course, Accommodation)
  ├── FAQIntegrator (Enrich with FAQ)
  │   └── FAQ Corpus (faq_corpus.json)
  └── StorageManager (Save results)
      ├── JSONExporter
      ├── CSVExporter
      └── DatabaseStorage
```

## Data Flow

```
URL → Smart Router → HTML/JS Scraper → Data Processor → FAQ Integrator → Storage
                          ↓
                    Detection of JS requirement
```

## Smart Routing Logic

The `SmartScraper.resolve_scraper()` decides which scraper to use in this order:

1. **Explicit override** — if `force_html` or `force_js` was passed, use it directly.
2. **Configuration hint** — if `institutions.yaml` declares `type: "javascript"` or `type: "html"`, honor it.
3. **Probe analysis** — if neither override nor config hint exists, fetch the page's initial HTML and score it:

   **JS Indicators (each adds points):**
   - Framework root elements: `#root`, `#app`, `#__next`, `#__nuxt`, `data-reactroot`
   - Framework globals: `__NEXT_DATA__`, `__NUXT__`, `window.__INITIAL_STATE__`, `ng-version`
   - Bundler artifacts: `main.[hash].js`, `chunk-vendors`, `.jsx`, `.tsx`
   - Sparse visible text (rendered content only after JS execution)

   **HTML Indicators (each adds points):**
   - Semantic content: `<h1>`, `<h2>`, `<article>`, `<table>`, `<li>` course items
   - Rich text content relative to total markup

   If **JS score > HTML score**, the JavaScriptScraper (Playwright) is used. Otherwise the HTMLScraper (requests + BeautifulSoup) is used.

## BaseScraper Responsibilities

- **Pagination**: appends `pagination_param` to the URL per page; detects next pages via `a[rel="next"]`, `.next`, `li.next a`, or pagination numbers greater than the current page.
- **Retry**: exponential backoff with configurable `MAX_RETRIES` and `RETRY_BACKOFF`. Special handling for HTTP 429 (rate limit) which backs off twice as long.
- **Rate limiting**: `delay` seconds between requests.
- **Selector extraction**: supports CSS pseudo-selectors:
  - `h2.course-title` → text content
  - `a.course-link::attr(href)` → attribute value
  - `.desc::text` → text content
  - `.desc::html` → inner HTML
  - `.tag::all` → list of matches
  - Relative URLs are resolved against the page URL.
- **Metadata**: `ScrapeMetadata` dataclass tracks start/end timestamps, item counts, error counts, pages scraped, final URL.

## JavaScriptScraper Details

- Lazy-launches a single headless Chromium per scraper instance.
- Waits for `wait_selector` (from YAML config) up to `WAIT_SELECTOR_TIMEOUT` ms.
- Falls back to a fixed 2s wait when no selector is configured.
- Captures a screenshot to `data/screenshots/` on timeout or JS error.

## DataProcessor Details

- Cleans whitespace, strips empty fields.
- Normalizes qualification levels (`BSc` → `bachelor`, `MSc` → `master`, etc.) and study modes (`Full-time` → `full_time`).
- Parses prices from strings like `£150 per week` → `150.0`.
- Validates with Pydantic; tracks validation/processing errors per record.
- Drops records lacking both title and URL.

## FAQIntegrator Details

- Loads `data/faq_corpus.json` (list of `{question, answer, keywords}`).
- Computes a hybrid relevance score (see `faq_integrator.py`):
  - 45% keyword coverage (fraction of the FAQ's configured keywords found in the record)
  - 25% question-token overlap
  - 15% answer-token overlap
  - 15% character-level SequenceMatcher similarity to the question
- Retains top-k FAQs above `FAQ_RELEVANCE_THRESHOLD` (default 0.10).

## Storage Details

- **JSONExporter**: writes `{metadata, data}` to `data/raw/` or `data/processed/`.
- **CSVExporter**: flattens records (nested fields JSON-serialized) to `data/processed/`.
- **DatabaseStorage**:
  - SQLite by default (stdlib `sqlite3`).
  - PostgreSQL if `DATABASE_URL` starts with `postgresql://` (requires `psycopg2-binary`).
  - Creates per-source tables (`<institution>_<data_type>`).
  - Stores generic columns (title, url, description, institution, data_type, scraped_at, related_faqs, extra JSON).
  - Provides `get_statistics()` for totals.

## Configuration Precedence

1. Command-line flags (`--force-js`, `--force-html`, `--delay`, `--max-pages`)
2. `.env` / environment variables
3. `config/institutions.yaml` (source config)
4. `config/selectors.yaml` (extraction config)
5. `config/settings.py` defaults
