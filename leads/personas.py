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

Two generators:
  * ``v1`` (default, legacy): labels come from the persona name (= rubric).
  * ``v2`` (opt-in, ``--generator v2``): inverted generator that fits
    ``P(counsellor_label | features)`` on the labelled real rows with a
    RandomForest, bootstraps a real profile, samples the label from
    ``predict_proba`` (not argmax) and then samples engagement conditioned on
    that label. Engagement features are excluded from the label model. This
    fixes the negative augmentation delta documented in ``leads/RESULTS.md``
    (v1 labels only agreed with the counsellor labels ~7% of the time).
    v2 is leakage-safe by default (``--bootstrap-pool train``): the label model
    is fit on, and synthesis bootstraps from, an 80/20 stratified *train* split
    of the labelled real rows only, so a real holdout drawn by the same split
    is never seen by the generator. The split is driven by ``--seed`` (the same
    seed drives split and generator in ``leads.eval_augmentation``); it used to
    be pinned to seed 42 whatever ``--seed`` said, so the documented "reproduce"
    command built a different pool than the evaluation. ``--group-split`` makes
    the split lead-grouped (no CRM id straddles train and holdout).

Usage:
    python -m leads.personas --n 1000 --seed 42 --out data/processed
    python -m leads.personas --n 1000 --seed 42 --generator v2
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


# --------------------------------------------------------------------------- #
# v2 -- inverted generator: sample labels from P(counsellor_label | features)
# --------------------------------------------------------------------------- #
# Rationale (leads/RESULTS.md, Finding 3): v1 labels come from the *persona*
# name, i.e. the rubric. The real counsellor labels come from a different
# labelling function (only ~7% agreement for the Cold class), so augmenting
# with v1 made real-world macro F1 WORSE. v2 inverts the dependency:
#
#   1. fit P(counsellor_label | profile features) on the labelled real rows
#      with a RandomForest (engagement features are excluded -- real leads
#      have no engagement, so it would be a synthetic-only confound);
#   2. bootstrap a real profile row;
#   3. sample the label from ``predict_proba`` (NOT argmax), so the synthetic
#      label follows the real labelling function rather than the rubric;
#   4. sample engagement features conditioned on the sampled label.
#
# Deterministic given ``seed``: the seed drives the bootstrap RNG, the label
# sampling and the RandomForest ``random_state``. v1 is untouched and remains
# byte-identical for a given seed (use ``--generator v1``).
GENERATORS = ("v1", "v2")

#: Label id -> display name (inverse of PERSONA_LABELS).
LABEL_NAMES = {v: k for k, v in PERSONA_LABELS.items()}

#: Profile (non-engagement) fields carried over from a bootstrapped real row.
PROFILE_FIELDS = [
    "passport_status", "has_english_test", "english_band",
    "funding_method_present", "funding_clarity", "destination_uk",
    "has_course", "has_intake", "qual_level", "study_gap_mentioned",
    "previous_application_mentioned", "note_word_count",
]


def _to_native(value):
    """numpy / NaN scalar -> plain, JSON-friendly Python value."""
    if value is None:
        return 0
    if isinstance(value, float) and math.isnan(value):
        return 0.0
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, AttributeError):  # pragma: no cover - defensive
            return value
    if isinstance(value, float):
        return round(value, 6)
    return value


def _prepare_real(frame):
    """Clean a real feature frame: map labels, drop unrated, ensure columns.

    Mirrors ``train_ml.align_schema`` but keeps the raw profile columns
    (including ``english_band``) so bootstrapped rows can be emitted verbatim.
    """
    import numpy as np
    import pandas as pd
    from leads.train_ml import CATEGORICAL, LABEL_MAP, NUMERIC, TARGET, VALID_LABELS

    if TARGET not in frame.columns:
        raise ValueError("real frame has no %r column" % TARGET)
    df = frame.copy()
    if not pd.api.types.is_integer_dtype(df[TARGET]):
        df[TARGET] = df[TARGET].map(
            lambda v: LABEL_MAP.get(str(v).strip().title(), np.nan)
            if not isinstance(v, (int, np.integer)) else v
        )
    df = df[df[TARGET].isin(VALID_LABELS)].copy()
    df[TARGET] = df[TARGET].astype(int)
    for col in list(CATEGORICAL) + list(NUMERIC) + PROFILE_FIELDS:
        if col not in df.columns:
            df[col] = np.nan
    return df.reset_index(drop=True)


class CounsellorLabelModel:
    """RandomForest estimate of ``P(counsellor_label | profile features)``.

    Fitted on the labelled real rows only. Engagement features are explicitly
    excluded from the feature matrix, so the label model can never learn the
    synthetic-only engagement confound documented in ``leads/RESULTS.md``.
    """

    def __init__(self, real_frame, n_estimators=500, random_state=42,
                 class_weight="balanced"):
        from sklearn.ensemble import RandomForestClassifier
        from leads.train_ml import ENGAGEMENT, build_matrix

        self.frame = _prepare_real(real_frame)
        if self.frame.empty:
            raise ValueError("no labelled real rows to fit the label model")
        X, y, feats = build_matrix(self.frame, no_engagement=True)
        for col in ENGAGEMENT:
            if any(f == col or f.startswith(col + "_") for f in feats):
                raise ValueError("engagement feature leaked into label model: %s" % col)
        self.X = X.reset_index(drop=True)
        self.feature_names = list(feats)
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, class_weight=class_weight,
            n_jobs=-1, random_state=random_state,
        )
        self.model.fit(self.X, y)
        self.classes = [int(c) for c in self.model.classes_]
        self.n_train = int(len(y))
        #: Cached P(label | features) for every real row. Computed once in a
        #: batch: single-row ``predict_proba`` calls are pathologically slow
        #: because scikit-learn re-enters joblib for every call.
        self._proba_cache = None
        #: Optional positional row indices into ``self.frame`` that the v2
        #: generator may bootstrap from. ``None`` = every fitted row. The
        #: leakage-safe protocol fits this model on the train split only, so
        #: its default pool already excludes the real holdout.
        self.bootstrap_indices = None

    def label_proba(self, row_index):
        """``P(label | features)`` for a bootstrapped real row."""
        if self._proba_cache is None:
            self._proba_cache = self.model.predict_proba(self.X)
        return [float(p) for p in self._proba_cache[row_index]]

    def sample_label(self, row_index, rng):
        """Draw the label from ``predict_proba`` -- deliberately not argmax."""
        return int(rng.choices(self.classes, weights=self.label_proba(row_index))[0])

    def source_row(self, row_index):
        return self.frame.iloc[row_index]


def generate_session_v2(session_id, rng, label_model):
    """One v2 session: bootstrap a real profile, sample label, then behaviour.

    The label is drawn from ``label_model.predict_proba`` on the bootstrapped
    row (not argmax). Engagement features are then drawn from the same
    persona-conditioned pools used by v1, but keyed on the *sampled counsellor
    label* instead of the rubric persona.
    """
    n_rows = len(label_model.frame)
    pool = getattr(label_model, "bootstrap_indices", None)
    if pool is None:
        pool = list(range(n_rows))
    idx = int(pool[rng.randrange(len(pool))])
    label = label_model.sample_label(idx, rng)
    src = label_model.source_row(idx)
    profile = {field: _to_native(src[field]) for field in PROFILE_FIELDS}
    persona = LABEL_NAMES[label]
    behaviour = _sample_behaviour(persona, rng)   # v1 helper, keyed by sampled label
    cats = behaviour.pop("question_categories")
    lead = score_row(profile)
    session = {
        "session_id": session_id,
        "persona": persona,
        "label": label,
        "question_categories": cats,
        "lead_score": lead.score,
        "lead_label": lead.label,
        "source_index": int(idx),
        "source_label": int(src["label"]),
    }
    if "row_id" in src.index:
        #: Global index into the full labelled real frame (set by the
        #: leakage-safe split) -- lets callers assert the holdout is excluded.
        session["source_row_id"] = int(_to_native(src["row_id"]))
    session.update(profile)
    session.update(behaviour)
    return session


def generate_sessions_v2(n=1000, seed=42, real_frame=None, label_model=None,
                         priors=None, bootstrap_pool="train", group_split=False):
    """Deterministic inverted-generator corpus.

    Parameters
    ----------
    real_frame : DataFrame, optional
        Real feature rows (output of ``leads.features``). Loaded from
        ``data/processed`` via ``train_ml.load_real`` when omitted.
    label_model : CounsellorLabelModel, optional
        Pre-fitted model; when supplied ``real_frame`` is ignored (used by the
        tests and by ``leads.eval_augmentation`` so they need no on-disk data).
        The bootstrap pool is then the model's own fitted frame.
    priors : dict, optional
        Accepted for CLI symmetry only -- v2 derives label frequencies from
        the real conditional, so persona priors are ignored (recorded in meta).
    bootstrap_pool : {"train", "all"}
        ``"train"`` (default, leakage-safe): fit the label model on an 80/20
        stratified train split (seed 42) of the labelled real rows and
        bootstrap profiles from those train rows only, so a real holdout drawn
        by the same split is never seen by the generator. ``"all"`` restores the
        legacy behaviour (fit and bootstrap on every labelled row) and is
        retained only to quantify the leak.
    group_split : bool
        With ``bootstrap_pool="train"``, split by CRM id (``crm_id`` column of
        ``real_frame``) so no lead straddles train and holdout.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if bootstrap_pool not in ("train", "all"):
        raise ValueError("bootstrap_pool must be 'train' or 'all'")
    if label_model is None:
        import numpy as np
        from leads.train_ml import split_real_indices
        if real_frame is None:
            from leads.train_ml import load_real
            real_frame = load_real()
        frame = _prepare_real(real_frame)
        if "row_id" not in frame.columns:
            frame["row_id"] = np.arange(len(frame), dtype=int)
        if bootstrap_pool == "train":
            # ``seed`` drives the protocol split, exactly as in
            # leads.eval_augmentation (one seed -> split + generators).
            groups = None
            if group_split:
                from leads.train_ml import lead_groups
                groups = lead_groups(frame)
            train_idx, _ = split_real_indices(
                frame, random_state=seed, groups=groups)
            frame = frame.iloc[train_idx].reset_index(drop=True)
        label_model = CounsellorLabelModel(frame, random_state=seed)
    rng = random.Random(seed)
    sessions = [
        generate_session_v2("synth-%05d" % i, rng, label_model)
        for i in range(n)
    ]
    label_counts = Counter(s["label"] for s in sessions)
    meta = {
        "generator": "v2",
        "n": n,
        "seed": seed,
        "priors": priors,
        "priors_ignored": priors is not None,
        "label_counts": {
            LABEL_NAMES[k]: int(label_counts.get(k, 0)) for k in LABEL_NAMES
        },
        "persona_counts": dict(Counter(s["persona"] for s in sessions)),
        "bootstrap_pool": {
            "mode": bootstrap_pool,
            "n_rows": int(len(label_model.bootstrap_indices)
                          if label_model.bootstrap_indices is not None
                          else len(label_model.frame)),
            "source_row_ids": sorted({int(s["source_row_id"])
                                      for s in sessions
                                      if "source_row_id" in s}),
        },
        "label_model": {
            "type": "RandomForestClassifier",
            "n_estimators": int(label_model.model.n_estimators),
            "class_weight": str(label_model.model.class_weight),
            "random_state": int(label_model.model.random_state),
            "n_train": int(label_model.n_train),
            "features": list(label_model.feature_names),
            "excludes_engagement": True,
        },
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
    parser.add_argument(
        "--generator", choices=list(GENERATORS), default="v1",
        help="v1 = legacy persona generator (default, byte-identical); "
             "v2 = inverted generator sampling labels from "
             "P(counsellor_label | features) fitted on the real rows",
    )
    parser.add_argument(
        "--group-split", action="store_true",
        help="v2 only: lead-grouped train split (no CRM id straddles)")
    parser.add_argument(
        "--bootstrap-pool", choices=["train", "all"], default="train",
        help="v2 only: 'train' (default) fits the label model and bootstraps "
             "profiles from the labelled train split only (leakage-safe); "
             "'all' uses every labelled row (legacy behaviour, leaks a real "
             "holdout drawn from the same pool).",
    )
    args = parser.parse_args(argv)
    priors = None
    if args.cold_frac is not None or args.warm_frac is not None \
            or args.hot_frac is not None:
        priors = {
            "Cold": args.cold_frac or 0.0,
            "Warm": args.warm_frac or 0.0,
            "Hot": args.hot_frac or 0.0,
        }

    if args.generator == "v2":
        if priors is not None:
            print("WARNING: --generator v2 ignores --cold/warm/hot-frac; "
                  "labels are sampled from P(counsellor_label | features).",
                  file=sys.stderr)
        sessions, meta = generate_sessions_v2(
            n=args.n, seed=args.seed, priors=priors,
            bootstrap_pool=args.bootstrap_pool, group_split=args.group_split)
        csv_path, json_path = write_outputs(sessions, meta, args.out)
        print("=" * 70)
        print("PERSONA SESSION SIMULATION v2 (inverted: real label function)")
        print("=" * 70)
        print("Sessions: %d (seed=%s)" % (len(sessions), meta["seed"]))
        print("Labels:   %s" % (meta["label_counts"],))
        lm = meta["label_model"]
        print("Label model: %s(%d trees, n_train=%d, engagement excluded=%s)"
              % (lm["type"], lm["n_estimators"], lm["n_train"],
                 lm["excludes_engagement"]))
        print("Bootstrap pool: %s (%d rows)"
              % (meta["bootstrap_pool"]["mode"],
                 meta["bootstrap_pool"]["n_rows"]))
        print("Saved:  %s" % csv_path)
        print("        %s" % json_path)
        return 0

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
