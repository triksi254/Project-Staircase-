"""Cold / Warm / Hot persona simulator: synthetic chatbot sessions.

Anchored to measured feature/label distributions from the real counsellor
corpus (leads.features + leads.rubric), for training the ML lead model and
the hybrid score alpha * rule + (1 - alpha) * ml.

Grounding priors (measured 2026-09-15, n=964 scored rows):
  rubric Cold 65 / Warm 395 / Hot 504
  counsellor Cold 374 / Good 426 / Excellent 61 / unrated 103

Caveats: english_band is noisy in the real extractor (years leak in as
bands, e.g. 2014.0); personas draw from realistic IELTS ranges instead.
note_word_count is 0.0 for every real row; session message/word counts are
persona-authored behavioural signals.

Deterministic given seed (stdlib random.Random only).

Usage:
    python -m leads.personas --n 1000 --seed 42 --out data/processed
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leads.features import (
    FUNDING_CLEAR,
    FUNDING_PARTIAL,
    FUNDING_UNCLEAR,
    FUNDING_UNKNOWN,
    PASSPORT_EXPIRED_NOT_RENEWED,
    PASSPORT_NONE,
    PASSPORT_VALID,
)
from leads.rubric import score_row

SESSION_COLUMNS = [
    "session_id", "persona", "label", "passport_status",
    "has_english_test", "english_band", "funding_method_present",
    "funding_clarity", "destination_uk", "has_course", "has_intake",
    "qual_level", "study_gap_mentioned", "previous_application_mentioned",
    "note_word_count", "message_count", "avg_delay_s",
    "question_category_entropy", "visa_intent_mentioned",
    "returning_session", "session_word_count", "lead_score", "lead_label",
]

PERSONA_LABELS = {"Cold": 0, "Warm": 1, "Hot": 2}

QUESTION_CATEGORIES = [
    "Entry Requirements", "Fees & Funding", "Scholarships",
    "Program Details", "Accommodation", "Visa & Immigration",
    "Application Process", "English Language", "General Enquiries",
]

BAND_POOLS = {
    "Cold": [0.0, 0.0, 0.0, 5.0, 5.5, 6.0],
    "Warm": [0.0, 5.5, 6.0, 6.0, 6.5, 6.5],
    "Hot": [6.0, 6.5, 6.5, 7.0, 7.0, 7.5, 8.0],
}

MEASURED_PRIORS = {"Cold": 65 / 964, "Warm": 395 / 964, "Hot": 504 / 964}


# Profile marginals measured from the real corpus (scratch/_persona_stats.py).
# Cold n=65: passport valid .69/none .31; funding unknown .57 unclear .14
# partial .29; qual unknown .94 bachelor .06; UK .23; course .28; intake .26;
# english .05; gap .15; prev-app .03.
# Warm n=395: passport valid .91/none .09; funding clear .21 partial .42
# unclear .27 unknown .10; qual unknown .66 bachelor .18 master .08
# diploma .08; UK .79; course .77; intake .71; english .26; gap .41; prev .10.
# Hot n=504: passport valid 1.0; funding clear .92 partial .08; qual
# bachelor .36 unknown .30 diploma .22 master .12; UK .99; course .99;
# intake .97; english .78; gap .48; prev-app .13.

def _sample_profile(persona, rng):
    r = rng.random
    if persona == "Cold":
        passport = PASSPORT_VALID if r() < 0.69 else PASSPORT_NONE
        funding = rng.choices(
            [FUNDING_UNKNOWN, FUNDING_UNCLEAR, FUNDING_PARTIAL],
            weights=[0.57, 0.14, 0.29])[0]
        qual = 3 if r() < 0.06 else 0
        dest = 1 if r() < 0.23 else 0
        course = 1 if r() < 0.28 else 0
        intake = 1 if r() < 0.26 else 0
        has_eng = 1 if r() < 0.05 else 0
        gap = 1 if r() < 0.15 else 0
        prev = 1 if r() < 0.03 else 0
    elif persona == "Warm":
        u = r()
        if u < 0.91:
            passport = PASSPORT_VALID
        elif u < 0.997:
            passport = PASSPORT_NONE
        else:
            passport = PASSPORT_EXPIRED_NOT_RENEWED
        funding = rng.choices(
            [FUNDING_CLEAR, FUNDING_PARTIAL, FUNDING_UNCLEAR, FUNDING_UNKNOWN],
            weights=[0.21, 0.42, 0.27, 0.10])[0]
        qual = rng.choices([0, 3, 4, 2], weights=[0.66, 0.18, 0.08, 0.08])[0]
        dest = 1 if r() < 0.79 else 0
        course = 1 if r() < 0.77 else 0
        intake = 1 if r() < 0.71 else 0
        has_eng = 1 if r() < 0.26 else 0
        gap = 1 if r() < 0.41 else 0
        prev = 1 if r() < 0.10 else 0
    else:
        passport = PASSPORT_VALID
        funding = FUNDING_CLEAR if r() < 0.92 else FUNDING_PARTIAL
        qual = rng.choices(
            [3, 0, 2, 4, 1, 5],
            weights=[0.36, 0.30, 0.22, 0.12, 0.005, 0.005])[0]
        dest = 1 if r() < 0.99 else 0
        course = 1 if r() < 0.99 else 0
        intake = 1 if r() < 0.97 else 0
        has_eng = 1 if r() < 0.78 else 0
        gap = 1 if r() < 0.48 else 0
        prev = 1 if r() < 0.13 else 0
    if has_eng:
        band = rng.choice(BAND_POOLS[persona])
        if band == 0.0:
            has_eng = 0
    else:
        band = 0.0
    funding_present = 0 if funding == FUNDING_UNKNOWN else 1
    return {
        "passport_status": passport, "has_english_test": has_eng,
        "english_band": band, "funding_method_present": funding_present,
        "funding_clarity": funding, "destination_uk": dest,
        "has_course": course, "has_intake": intake, "qual_level": qual,
        "study_gap_mentioned": gap, "previous_application_mentioned": prev,
        "note_word_count": 0,
    }

# Behavioural chat signals. Hot = engaged (many messages, short delays,
# high category entropy, visa intent, often returning). Cold = reverse.

def _category_entropy(n_msg, cats, rng):
    if n_msg <= 1 or len(cats) <= 1:
        return 0.0
    counts = [0] * len(cats)
    for _ in range(n_msg):
        idx = 0 if rng.random() < 0.5 else rng.randrange(len(cats))
        counts[idx] += 1
    return round(-sum(
        (c / n_msg) * math.log2(c / n_msg) for c in counts if c), 4)


def _sample_behaviour(persona, rng):
    if persona == "Cold":
        n_msg = rng.randint(1, 3)
        delay = round(rng.uniform(120.0, 600.0), 1)
        n_cats = 1
        visa = 1 if rng.random() < 0.05 else 0
        returning = 1 if rng.random() < 0.05 else 0
        words = rng.randint(5, 30)
    elif persona == "Warm":
        n_msg = rng.randint(3, 7)
        delay = round(rng.uniform(30.0, 180.0), 1)
        n_cats = rng.randint(2, 4)
        visa = 1 if rng.random() < 0.35 else 0
        returning = 1 if rng.random() < 0.30 else 0
        words = rng.randint(30, 120)
    else:
        n_msg = rng.randint(6, 14)
        delay = round(rng.uniform(8.0, 60.0), 1)
        n_cats = rng.randint(3, 6)
        visa = 1 if rng.random() < 0.75 else 0
        returning = 1 if rng.random() < 0.60 else 0
        words = rng.randint(100, 350)
    cats = rng.sample(QUESTION_CATEGORIES, k=min(n_cats, len(QUESTION_CATEGORIES)))
    return {
        "message_count": n_msg, "avg_delay_s": delay,
        "question_category_entropy": _category_entropy(n_msg, cats, rng),
        "visa_intent_mentioned": visa, "returning_session": returning,
        "session_word_count": words, "question_categories": cats,
    }


def generate_session(session_id, persona, rng):
    profile = _sample_profile(persona, rng)
    behaviour = _sample_behaviour(persona, rng)
    cats = behaviour.pop("question_categories")
    lead = score_row(profile)
    session = {
        "session_id": session_id, "persona": persona,
        "label": PERSONA_LABELS[persona], "question_categories": cats,
        "lead_score": lead.score, "lead_label": lead.label,
    }
    session.update(profile)
    session.update(behaviour)
    return session


def generate_sessions(n=1000, seed=42, priors=None):
    if n <= 0:
        raise ValueError("n must be positive")
    priors = dict(priors) if priors else dict(MEASURED_PRIORS)
    total = sum(priors.values())
    if total <= 0:
        raise ValueError("priors must sum to a positive value")
    personas = list(priors)
    weights = [priors[p] / total for p in personas]
    rng = random.Random(seed)
    sessions = [
        generate_session("synth-%05d" % i,
                         rng.choices(personas, weights=weights)[0], rng)
        for i in range(n)
    ]
    meta = {
        "n": n, "seed": seed, "priors": priors,
        "persona_counts": dict(Counter(s["persona"] for s in sessions)),
        "rubric_label_counts": dict(Counter(s["lead_label"] for s in sessions)),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    return sessions, meta


def write_outputs(sessions, meta, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / ("synthetic_sessions_%d_%s.csv" % (len(sessions), ts))
    json_path = out_dir / ("synthetic_sessions_%d_%s.json" % (len(sessions), ts))
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fh:
        rows = []
        for s in sessions:
            row = {k: s.get(k) for k in SESSION_COLUMNS}
            rows.append(row)
        writer = csv.DictWriter(fh, fieldnames=SESSION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    payload = {"metadata": meta, "columns": SESSION_COLUMNS, "data": sessions}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return csv_path, json_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Persona session simulator")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--cold-frac", type=float, default=None)
    parser.add_argument("--warm-frac", type=float, default=None)
    parser.add_argument("--hot-frac", type=float, default=None)
    args = parser.parse_args(argv)
    priors = None
    if args.cold_frac is not None or args.warm_frac is not None \
            or args.hot_frac is not None:
        priors = {
            "Cold": args.cold_frac or 0.0,
            "Warm": args.warm_frac or 0.0,
            "Hot": args.hot_frac or 0.0,
        }
    sessions, meta = generate_sessions(n=args.n, seed=args.seed, priors=priors)
    csv_path, json_path = write_outputs(sessions, meta, args.out)
    print("=" * 70)
    print("PERSONA SESSION SIMULATION (Cold / Warm / Hot)")
    print("=" * 70)
    print("Sessions: %d (seed=%s)" % (len(sessions), meta["seed"]))
    print("Personas: %s" % (meta["persona_counts"],))
    print("Rubric:   %s" % (meta["rubric_label_counts"],))
    print("Saved:  %s" % csv_path)
    print("        %s" % json_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
