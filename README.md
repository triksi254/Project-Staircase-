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
├── main.py                      # Entry point
├── requirements.txt             # Dependencies
├── .env.example                 # Environment template
├── batch_config_example.json    # Batch job config
│
├── config/
│   ├── settings.py              # Global configuration (env-based)
│   ├── institutions.yaml        # Institution definitions
│   └── selectors.yaml           # CSS selectors
│
├── models/
│   └── schemas.py               # Pydantic Course/Accommodation models
│
├── scrapers/
│   ├── base_scraper.py          # Abstract base class
│   ├── html_scraper.py          # BeautifulSoup-based
│   ├── js_scraper.py            # Playwright-based
│   └── smart_scraper.py         # Router (JS detection)
│
├── processors/
│   └── data_processor.py        # Data cleaning & validation
│
├── storage/
│   └── storage.py               # Database & export
│
├── faq/
│   └── faq_integrator.py        # FAQ cross-reference
│
├── data/
│   ├── raw/                     # Downloaded raw JSON
│   ├── processed/               # Cleaned data (CSV/JSON)
│   └── faq_corpus.json          # FAQ database
│
├── logs/
│   └── scraper.log              # Application logs
│
└── tests/
    ├── test_scrapers.py
    └── test_processors.py
```

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
