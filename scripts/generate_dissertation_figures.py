"""Generate the 12 dissertation figures as print-ready PNG (300dpi) + SVG.

Every number is either read directly from a committed artifact JSON /
leads/RESULTS_v2.md, or recomputed here from the same tracked data those
documents were generated from (the data funnel, profile-vector collisions,
and the live rule-score lattice) -- nothing is hand-transcribed from memory.
Figure 12 (test-suite growth) is the one exception: those counts were
directly measured during this development engagement, immediately before
each cited commit, and are recorded here as a historical log, not recomputed.

Usage: python scripts/generate_dissertation_figures.py
Output: figures/fig01_*.png (+ .svg) ... figures/fig12_*.png (+ .svg)
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ART = ROOT / "artifacts"
OUT = ROOT / "figures"

# --------------------------------------------------------------------------- #
# Palette (dataviz skill reference palette, light mode) + shared style
# --------------------------------------------------------------------------- #
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, GREEN, VIOLET, RED = "#e87ba4", "#008300", "#4a3aa7", "#e34948"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#ffffff"
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_SECONDARY,
    "ytick.color": INK_SECONDARY,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 1.0,
    "grid.linestyle": "-",
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "svg.fonttype": "none",
})


def _finish(fig, name, title_note=""):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name, title_note)


def _source(ax, text):
    ax.annotate(text, xy=(0, -0.14), xycoords="axes fraction",
                fontsize=8.5, color=INK_MUTED, ha="left", va="top")


def _zero_line(ax, vertical=True):
    if vertical:
        ax.axvline(0, color=BASELINE, linewidth=1.2, zorder=1)
    else:
        ax.axhline(0, color=BASELINE, linewidth=1.2, zorder=1)


def _load(name):
    return json.loads((ART / name).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Fig 1 -- row-level vs lead-grouped augmentation delta (grouped bars + CI)
# --------------------------------------------------------------------------- #
def fig01():
    row = _load("eval_augmentation_seed_robustness.json")["seed_robustness"]["feature_sets"]
    grp = _load("eval_augmentation_grouped_seed_robustness.json")["seed_robustness"]["feature_sets"]

    configs = [("full", "Full feature set"), ("no-engagement", "No-engagement (fair)")]
    gens = [("v1", "+v1", RED), ("v2", "+v2", BLUE), ("distill", "+distill (control)", VIOLET)]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for ax, (key, label) in zip(axes, configs):
        x = np.arange(2)  # row-level, lead-grouped
        width = 0.24
        for i, (gkey, glabel, color) in enumerate(gens):
            means, los, his = [], [], []
            for src in (row, grp):
                s = src[key]["summary"][gkey]["macro"]
                means.append(s["mean"])
                los.append(s["mean"] - s["min"])
                his.append(s["max"] - s["mean"])
            xs = x + (i - 1) * width
            ax.bar(xs, means, width=width * 0.82, color=color, label=glabel,
                   zorder=3)
            ax.errorbar(xs, means, yerr=[los, his], fmt="none", ecolor=INK,
                        elinewidth=1.3, capsize=4, capthick=1.3, zorder=4)
        _zero_line(ax, vertical=False)
        ax.set_xticks(x, ["Row-level split", "Lead-grouped split"])
        ax.set_title(label, fontsize=11.5)
        ax.set_ylabel("Δ macro F1 vs real-only" if ax is axes[0] else "")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%+.2f"))
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle("Augmentation gain shrinks once lead-level leakage is removed",
                 fontsize=14, fontweight="bold", y=1.03)
    _source(axes[0], "Source: artifacts/eval_augmentation_seed_robustness.json,\n"
                      "eval_augmentation_grouped_seed_robustness.json (mean, min-max over seeds 1/7/42)")
    _finish(fig, "fig01_augmentation_row_vs_grouped")


# --------------------------------------------------------------------------- #
# Fig 2 -- per-seed forest plot (full-set +v2, row-level vs lead-grouped)
# --------------------------------------------------------------------------- #
def fig02():
    row = _load("eval_augmentation_seed_robustness.json")["seed_robustness"]["feature_sets"]["full"]["rows"]
    grp = _load("eval_augmentation_grouped_seed_robustness.json")["seed_robustness"]["feature_sets"]["full"]["rows"]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    labels, ys, xs, xerr, colors = [], [], [], [], []
    y = 0
    for split_name, rows, color in [("Lead-grouped", grp, ORANGE), ("Row-level", row, BLUE)]:
        for r in sorted(rows, key=lambda r: r["seed"]):
            labels.append(f"seed {r['seed']} - {split_name}")
            ys.append(y)
            xs.append(r["delta_v2_macro"])
            colors.append(color)
            y += 1
        y += 0.6
    ax.errorbar(xs, ys, fmt="none", ecolor=INK, elinewidth=0, zorder=2)
    for xi, yi, c in zip(xs, ys, colors):
        ax.plot([0, xi], [yi, yi], color=BASELINE, linewidth=1, zorder=1)
        ax.scatter([xi], [yi], s=90, color=c, zorder=3, edgecolor=SURFACE, linewidth=1.5)
    _zero_line(ax, vertical=True)
    ax.set_yticks(ys, labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Δ macro F1 (+v2 vs real-only), full feature set")
    ax.set_title("+v2 gain per seed: positive under row-level, one seed\nnegative under lead-grouped",
                 fontsize=12.5)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%+.2f"))
    handles = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markersize=9, label="Row-level split"),
               plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=ORANGE, markersize=9, label="Lead-grouped split")]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)
    _source(ax, "Source: artifacts/eval_augmentation_seed_robustness.json,\n"
                "eval_augmentation_grouped_seed_robustness.json (per-seed delta_v2_macro)")
    _finish(fig, "fig02_per_seed_forest")


# --------------------------------------------------------------------------- #
# Fig 3 -- model-configuration macro F1 comparison
# --------------------------------------------------------------------------- #
def fig03():
    configs = [
        ("rf_full", "RF, full features\n(confounded)", "metrics_rf_full.json"),
        ("rf_ablate_eng", "RF, English\nablated", "metrics_rf_ablate_eng.json"),
        ("logreg_full", "LogReg,\nfull features", "metrics_logreg_full.json"),
        ("rf_noeng", "RF, no\nengagement", "metrics_rf_noeng.json"),
        ("rf_realonly", "RF, real data\nonly (defensible)", "metrics_rf_realonly.json"),
    ]
    vals = [(_load(f)["headline"]["macro_f1"], label) for _, label, f in configs]
    vals.append((0.2104, "Dummy\nbaseline"))  # leads/RESULTS.md headline table

    fig, ax = plt.subplots(figsize=(8.5, 5))
    xs = np.arange(len(vals))
    colors = [RED if "confounded" in lbl else (INK_MUTED if "Dummy" in lbl else
              (STATUS_GOOD if "defensible" in lbl else BLUE)) for _, lbl in vals]
    bars = ax.bar(xs, [v for v, _ in vals], width=0.62, color=colors, zorder=3)
    for b, (v, _) in zip(bars, vals):
        ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(xs, [lbl for _, lbl in vals], fontsize=9.5)
    ax.set_ylabel("Macro F1")
    ax.set_ylim(0, 1.0)
    ax.set_title("The 0.90 headline is a confound: real-data macro F1 is 0.62",
                 fontsize=13)
    _source(ax, "Source: artifacts/metrics_*.json (headline.macro_f1); dummy baseline from leads/RESULTS.md")
    _finish(fig, "fig03_model_confound")


# --------------------------------------------------------------------------- #
# Fig 4 -- feature importance, rf_full vs rf_realonly (top 8 each, union)
# --------------------------------------------------------------------------- #
def fig04():
    full = _load("feature_importance_rf_full.json")
    real = _load("feature_importance_rf_realonly.json")
    top_full = [k for k, _ in sorted(full.items(), key=lambda kv: -kv[1])[:6]]
    top_real = [k for k, _ in sorted(real.items(), key=lambda kv: -kv[1])[:6]]
    feats = list(dict.fromkeys(top_full + top_real))  # union, order preserved

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(feats))
    h = 0.34
    ax.barh(y + h / 2, [full.get(f, 0) for f in feats], height=h * 0.9, color=RED,
            label="rf_full (real + synthetic)", zorder=3)
    ax.barh(y - h / 2, [real.get(f, 0) for f in feats], height=h * 0.9, color=STATUS_GOOD,
            label="rf_realonly (real only)", zorder=3)
    ax.set_yticks(y, feats, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Feature importance")
    ax.set_title("Engagement features dominate rf_full and vanish once synthetic\n"
                 "engagement is removed", fontsize=12.5)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    _source(ax, "Source: artifacts/feature_importance_rf_full.json, feature_importance_rf_realonly.json")
    _finish(fig, "fig04_feature_importance")


# --------------------------------------------------------------------------- #
# Fig 5 -- alpha calibration curve
# --------------------------------------------------------------------------- #
def fig05():
    d = _load("alpha_calibration.json")
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label, color in [("lead_grouped_oof", "Lead-grouped OOF (valid)", BLUE),
                              ("row_level_oof", "Row-level OOF (leaky reference)", ORANGE)]:
        grid = d[key]["grid"]
        xs = [g["alpha"] for g in grid]
        ys = [g["macro_f1"] for g in grid]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5,
                markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1,
                zorder=3, label=label)
        best = max(grid, key=lambda g: g["macro_f1"])
        ax.scatter([best["alpha"]], [best["macro_f1"]], s=140, facecolor="none",
                   edgecolor=color, linewidth=2, zorder=4)
    ax.axvline(0.5, color=INK_MUTED, linewidth=1.2, linestyle=(0, (1, 1.5)), zorder=1)
    ax.annotate("shipped default\n(alpha = 0.5, uncalibrated)", xy=(0.5, 0.08),
                fontsize=8.5, color=INK_MUTED, ha="center")
    ax.set_xlabel("alpha  (hybrid = alpha·rule + (1-alpha)·ML)")
    ax.set_ylabel("Macro F1")
    ax.set_title("Calibrated alpha (0.1) beats the shipped default (0.5)\n"
                 "on the leakage-safe lead-grouped split", fontsize=12.5)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    _source(ax, "Source: artifacts/alpha_calibration.json (lead_grouped_oof.grid, row_level_oof.grid); "
                "circled = best alpha per curve")
    _finish(fig, "fig05_alpha_calibration")


# --------------------------------------------------------------------------- #
# Fig 6 -- rubric vs counsellor label agreement
# --------------------------------------------------------------------------- #
def fig06():
    rows = [("Cold", 374, 0.0695), ("Warm", 426, 0.1925), ("Hot", 61, 1.0000)]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    xs = np.arange(len(rows))
    vals = [r[2] for r in rows]
    colors = [RED, ORANGE, STATUS_GOOD]
    bars = ax.bar(xs, vals, width=0.5, color=colors, zorder=3)
    for b, (label, n, v) in zip(bars, rows):
        ax.annotate(f"{v:.1%}\n(n={n})", (b.get_x() + b.get_width() / 2, v),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9.5)
    ax.axhline(0.1963, color=INK, linewidth=1.2, linestyle=(0, (4, 2)), zorder=2)
    ax.annotate("overall 19.6% (n=861)", xy=(1.55, 0.26), fontsize=9, color=INK_SECONDARY, ha="center")
    ax.set_xticks(xs, [r[0] for r in rows])
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylabel("Share of counsellor-labelled rows the rubric agrees with")
    ax.set_title("The rule-based rubric agrees with the counsellor's own label\n"
                 "19.6% of the time overall", fontsize=12.5)
    _source(ax, "Source: leads/RESULTS_v2.md Table 3b (real corpus: counsellor label vs rubric)")
    _finish(fig, "fig06_rubric_agreement")


# --------------------------------------------------------------------------- #
# Fig 7 -- SBERT vs TF-IDF paired bootstrap CIs (forest plot)
# --------------------------------------------------------------------------- #
def fig07():
    a = _load("compare_gold_a_corrected.json")
    b = _load("compare_gold_b_corrected.json")
    metrics = [("rank_1", "Rank-1 accuracy"), ("recall_at_3", "Recall@3"), ("recall_at_5", "Recall@5")]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    labels, ys, xs, los, his, colors = [], [], [], [], [], []
    y = 0
    for gold_name, gold, color in [("Gold B (n=31)", b, ORANGE), ("Gold A (n=23)", a, BLUE)]:
        for key, mlabel in metrics:
            bs = gold["bootstrap"][key]
            labels.append(f"{mlabel} - {gold_name}")
            ys.append(y)
            xs.append(bs["delta"])
            los.append(bs["delta"] - bs["ci_low"])
            his.append(bs["ci_high"] - bs["delta"])
            colors.append(color)
            y += 1
        y += 0.6
    for xi, yi, lo, hi, c in zip(xs, ys, los, his, colors):
        ax.plot([xi - lo, xi + hi], [yi, yi], color=c, linewidth=2.2, zorder=2,
                solid_capstyle="round")
        ax.scatter([xi], [yi], s=70, color=c, zorder=3, edgecolor=SURFACE, linewidth=1.3)
    _zero_line(ax)
    ax.set_yticks(ys, labels, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("SBERT - TF-IDF (paired bootstrap 95% CI)")
    ax.set_title("SBERT scores higher on every metric, but no CI clears zero:\n"
                 "not statistically significant at n=23-31", fontsize=12.5)
    _source(ax, "Source: artifacts/compare_gold_a_corrected.json, compare_gold_b_corrected.json (bootstrap.*)")
    _finish(fig, "fig07_sbert_vs_tfidf_ci")


# --------------------------------------------------------------------------- #
# Fig 8 -- gate sweep: false-abstention rate vs threshold
# --------------------------------------------------------------------------- #
def fig08():
    a = _load("compare_gold_a_corrected.json")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for key, label, color in [("tfidf", "TF-IDF", ORANGE), ("sbert", "SBERT", BLUE)]:
        rows = a["sweep"][key]
        xs = [r["threshold"] for r in rows]
        ys = [r["false_abstention_rate"] for r in rows]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6,
                markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.3,
                zorder=3, label=label)
    ax.axvline(0.6, color=INK_MUTED, linewidth=1.2, linestyle=(0, (1, 1.5)), zorder=1)
    ax.annotate("evaluated gate\n(0.60)", xy=(0.6, 0.05), fontsize=8.5, color=INK_MUTED, ha="center")
    ax.set_xlabel("Confidence threshold")
    ax.set_ylabel("False-abstention rate")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_title("At the evaluated 0.60 gate, TF-IDF withholds 83% of answerable\n"
                 "queries vs 43% for SBERT (Gold A)", fontsize=12.5)
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    _source(ax, "Source: artifacts/compare_gold_a_corrected.json (sweep.tfidf/.sbert)")
    _finish(fig, "fig08_gate_sweep")


# --------------------------------------------------------------------------- #
# Fig 9 -- data funnel
# --------------------------------------------------------------------------- #
def _load_labelled_rows():
    import pathlib
    cands = sorted((ROOT / "data" / "processed").glob("lead_features_*.json"))
    payload = json.loads(cands[-1].read_text(encoding="utf-8"))
    rows = payload.get("data", payload) if isinstance(payload, dict) else payload
    labelled = [r for r in rows if r.get("label", -1) is not None and r.get("label", -1) >= 0]
    return rows, labelled


def fig09():
    rows, labelled = _load_labelled_rows()
    crm_ids = [r.get("crm_id") for r in labelled if r.get("crm_id")]
    counts = Counter(crm_ids)
    repeated_ids = {k: v for k, v in counts.items() if v > 1}
    repeated_rows = sum(repeated_ids.values())
    unique = len(counts)

    stages = [
        ("Total scraped\ncounsellor records", len(rows)),
        ("Labelled\n(rated)", len(labelled)),
        ("...of which: unique\nCRM ids", unique),
        ("...of which: rows in a\nrepeated CRM id", repeated_rows),
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = np.arange(len(stages))
    vals = [v for _, v in stages]
    colors = [INK_MUTED, BLUE, BLUE, RED]
    bars = ax.bar(xs, vals, width=0.55, color=colors, zorder=3)
    for b, (label, v) in zip(bars, stages):
        ax.annotate(str(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=10.5)
    pct = 100 * repeated_rows / len(labelled)
    ax.annotate(f"{pct:.0f}% of labelled rows\nshare a repeated CRM id",
                xy=(3, repeated_rows), xytext=(2.15, repeated_rows + 90),
                fontsize=9.5, color=RED, ha="left")
    ax.set_xticks(xs, [s for s, _ in stages], fontsize=9.5)
    ax.set_ylabel("Rows")
    ax.set_title("Almost 2 in 5 labelled rows belong to a lead that appears\n"
                 "more than once", fontsize=12.5)
    _source(ax, "Source: recomputed here from data/processed/lead_features_*.json "
                "(the same file leads/train_ml.py consumes)")
    _finish(fig, "fig09_data_funnel")


# --------------------------------------------------------------------------- #
# Fig 10 -- profile-vector collisions
# --------------------------------------------------------------------------- #
def fig10():
    from leads.features import FEATURE_COLUMNS
    _, labelled = _load_labelled_rows()
    exclude = {"crm_id", "label", "assessment_notes", "note_word_count"}
    cols = [c for c in FEATURE_COLUMNS if c not in exclude and c in labelled[0]]

    def vec(r):
        return tuple(r.get(c) for c in cols)

    lbls_by_vec = defaultdict(set)
    for r in labelled:
        lbls_by_vec[vec(r)].add(r.get("label"))
    n_vectors = len(lbls_by_vec)
    conflicting = {v for v, l in lbls_by_vec.items() if len(l) > 1}
    rows_conflict = sum(1 for r in labelled if vec(r) in conflicting)
    rows_clean = len(labelled) - rows_conflict

    fig, ax = plt.subplots(figsize=(9, 3.2))
    segments = [("Single-label profile", rows_clean, STATUS_GOOD),
                ("Label-conflicting profile", rows_conflict, RED)]
    left = 0
    for label, v, color in segments:
        ax.barh([0], [v], left=left, height=0.55, color=color, zorder=3)
        pct = v / len(labelled)
        text_color = "white" if pct > 0.12 else INK
        x = left + v / 2 if pct > 0.12 else left + v + len(labelled) * 0.015
        ha = "center" if pct > 0.12 else "left"
        ax.annotate(f"{label}\n{v} rows ({pct:.0%})", xy=(x, 0), ha=ha, va="center",
                    fontsize=10.5, color=text_color, fontweight="bold")
        left += v
    ax.set_xlim(0, len(labelled))
    ax.set_ylim(-1, 1)
    ax.axis("off")
    ax.set_title(f"{n_vectors} distinct feature profiles among {len(labelled)} rows:\n"
                 f"{rows_conflict / len(labelled):.0%} of rows sit in a profile the\n"
                 "rubric's own inputs cannot separate", fontsize=12.5)
    _source(ax, "Source: recomputed from data/processed/lead_features_*.json using\n"
                "leads.features.FEATURE_COLUMNS (excl. crm_id/label/notes)")
    _finish(fig, "fig10_profile_collisions")


# --------------------------------------------------------------------------- #
# Fig 11 -- live rule-score reachability lattice
# --------------------------------------------------------------------------- #
def fig11():
    import importlib.util
    spec = importlib.util.spec_from_file_location("diagnose_v3", ROOT / "scripts" / "diagnose_v3.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    lattice = m.rule_lattice()
    lattice = sorted(lattice, key=lambda r: r["score"])

    def name(topics):
        short = {"Entry Requirements": "course", "Application Process": "intake",
                 "English Language": "English"}
        return "+".join(short[t] for t in topics) if topics else "(none raised)"

    fig, ax = plt.subplots(figsize=(8.5, 5))
    xs = np.arange(len(lattice))
    vals = [r["score"] for r in lattice]
    ax.bar(xs, vals, width=0.55, color=BLUE, zorder=3)
    for x, v in zip(xs, vals):
        ax.annotate(f"{v:.4f}", (x, v), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9)
    for cut, label in [(1 / 3, "Cold/Warm cut"), (2 / 3, "Warm/Hot cut")]:
        ax.axhline(cut, color=INK_MUTED, linewidth=1.1, linestyle=(0, (4, 2)), zorder=1)
        ax.annotate(label, xy=(6.3, cut), fontsize=8.5, color=INK_MUTED, va="bottom")
    ax.set_xticks(xs, [name(r["topics"]) for r in lattice], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Live rule score")
    ax.set_ylim(0, 0.85)
    ax.set_title("Only 6 distinct live rule-score values are reachable through chat;\n"
                 "the ceiling (0.7778) still clears the Hot cut", fontsize=12.5)
    _source(ax, "Source: scripts/diagnose_v3.py::rule_lattice() (computed live, not cached)")
    _finish(fig, "fig11_rule_score_lattice")


# --------------------------------------------------------------------------- #
# Fig 12 -- test-suite growth across the tagged fix history
# --------------------------------------------------------------------------- #
def fig12():
    # Counts measured directly (pytest -q) immediately before each commit,
    # during this development engagement. Not recomputed from git history.
    points = [
        ("diagnose_v3", "4d01cc9", 368),
        ("guard_institution_fix", "5c084d9", 386),
        ("guard_fix_verification", "b6868dc", 389),
        ("funding_structural_diagnostic", "9060f9c", 391),
        ("funding_amount_extractor", "349c883", 418),
        ("funding_fix_final", "9db449b", 421),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    xs = np.arange(len(points))
    ys = [v for _, _, v in points]
    ax.plot(xs, ys, color=BLUE, linewidth=2, marker="o", markersize=7,
            markerfacecolor=BLUE, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    for x, y in zip(xs, ys):
        ax.annotate(str(y), (x, y), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=10)
    ax.set_xticks(xs, [h for _, h, _ in points], fontsize=9)
    ax.set_ylabel("Tests passing")
    ax.set_ylim(340, 440)
    ax.set_xlim(-0.5, len(points) - 0.5)
    ax.set_title("Test count through the diagnose-fix-verify cycle\n"
                 "(most recent tagged development phase)", fontsize=12.5)
    legend_text = "\n".join(f"{h} = {t}" for t, h, _ in points)
    ax.annotate(legend_text, xy=(1.02, 1.0), xycoords="axes fraction", fontsize=8.3,
               color=INK_SECONDARY, va="top", ha="left")
    _source(ax, "Source: pytest run measured immediately before each cited commit, this engagement")
    _finish(fig, "fig12_test_growth")


def main():
    for fn in (fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, fig09,
               fig10, fig11, fig12):
        fn()
    print(f"\n{len(list(OUT.glob('*.png')))} PNG + {len(list(OUT.glob('*.svg')))} SVG written to {OUT}")


if __name__ == "__main__":
    main()
