"""End-to-end demo: query -> retrieval -> category -> hybrid score -> response.

Usage:
    python -m chatbot.demo "Do I need a visa?" --rule 0.8
    python -m chatbot.demo "How much is accommodation at Aston?" --rule 0.4
    python -m chatbot.demo "Can I get in with a KCSE C+?" --rule 0.6 --ml-proba 0.2,0.5,0.3
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chatbot.responder import ABSTAIN_MESSAGE as ABSTAIN

BAR = "=" * 64
RULE = "-" * 64


def _parse_ml_proba(raw: Optional[str]) -> Optional[List[float]]:
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 3:
        raise ValueError("--ml-proba must be P(Cold),P(Warm),P(Hot)")
    try:
        vals = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError("--ml-proba must be numeric") from exc
    if any(v < 0 for v in vals) or sum(vals) <= 0:
        raise ValueError("--ml-proba must be non-negative with positive sum")
    return vals


def _default_alpha() -> float:
    import logging
    try:
        from leads.hybrid import load_artifacts
        logging.disable(logging.CRITICAL)
        try:
            arts = load_artifacts()
        finally:
            logging.disable(logging.NOTSET)
        cfg = arts.get("config") or {}
        alpha = cfg.get("alpha", 0.5)
        return float(alpha)
    except Exception:
        return 0.5


def run_demo(
    query: str,
    rule: float = 0.5,
    ml_proba: Optional[List[float]] = None,
    alpha: Optional[float] = None,
    k: int = 3,
) -> Dict[str, Any]:
    from chatbot.responder import EscalationPolicy, respond
    from chatbot.retriever import Retriever, load_corpus
    from leads.hybrid import hybrid_score, ml_proba_to_score, score_to_label

    if alpha is None:
        alpha = _default_alpha()
    ml_score: Optional[float] = None
    if ml_proba is not None:
        ml_score = ml_proba_to_score(ml_proba)
    retriever = Retriever(load_corpus())
    policy = EscalationPolicy(alpha=alpha)
    # Pass the pre-combined hybrid as rule_score with alpha=1 so respond()
    # does not re-blend; escalation/labels still come from score_to_label.
    pre = hybrid_score(rule, ml_score, alpha)
    result = respond(query, retriever, rule_score=pre, ml_score=None,
                     policy=EscalationPolicy(min_confidence=policy.min_confidence,
                                             hot_threshold=policy.hot_threshold,
                                             alpha=1.0), top_k=k)
    label = score_to_label(float(result.get("lead_score", pre)))
    cands = result.get("candidates", [])[:k]
    resp = {
        "answer": result.get("answer", ""),
        "category": result.get("category") or "General Enquiries",
        "institution": result.get("institution") or "General",
        "confidence": float(result.get("confidence", 0.0)),
        "fallback": bool(result.get("answer", "") == ABSTAIN) or result.get("cited_question") is None,
        "source_id": None,
    }
    questions = [e.question for e in retriever.entries] if hasattr(retriever, "entries") else []
    if result.get("cited_question") in questions:
        resp["source_id"] = questions.index(result["cited_question"])
    return {
        "query": query,
        "rule": rule,
        "ml_proba": list(ml_proba) if ml_proba is not None else None,
        "alpha": alpha,
        "ml_score": ml_score,
        "hybrid_score": float(result.get("lead_score", pre)),
        "label": label,
        "escalate": bool(result.get("escalate", False)),
        "priority": result.get("priority"),
        "response": resp,
        "candidates": [
            {"score": float(c.get("score", 0.0)), "question": c.get("question", "")}
            for c in cands
        ],
    }


def format_human(out: Dict[str, Any], threshold: float = 0.60) -> str:
    ml_txt = "rule-only" if out["ml_proba"] is None else "ml-blend"
    lines = [
        BAR,
        f"  Query:    {out['query']}",
        f"  Rule:     {out['rule']:.2f}    ML: {ml_txt}    alpha: {out['alpha']:.2f}",
        RULE,
        "  Response:",
        f"    {out['response']['answer']}",
        f"  [grounded: entry #{out['response']['source_id']}]",
        "",
        f"  Category:       {out['response']['category']}",
        f"  Institution:    {out['response']['institution']}",
        f"  Confidence:     {out['response']['confidence']:.2f}   (threshold {threshold:.2f})",
        f"  Fallback:       {'yes' if out['response']['fallback'] else 'no'}",
        "",
        f"  Hybrid score:   {out['hybrid_score']:.2f}   -> {out['label']}",
        f"  Escalation:     {'yes' if out['escalate'] else 'no'}",
        RULE,
        f"  Top-{len(out['candidates'])} candidates:",
    ]
    for i, c in enumerate(out["candidates"], 1):
        lines.append(f"    {i}. {c['score']:.2f}  {c['question']}")
    lines.append(BAR)
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Chatbot end-to-end demo")
    ap.add_argument("query")
    ap.add_argument("--rule", type=float, default=0.5)
    ap.add_argument("--ml-proba", type=str, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    if not 0.0 <= args.rule <= 1.0:
        raise SystemExit("--rule must be in [0, 1]")
    ml_proba = _parse_ml_proba(args.ml_proba)
    out = run_demo(args.query, rule=args.rule, ml_proba=ml_proba,
                   alpha=args.alpha, k=args.k)
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(format_human(out))
    return 0


def run_cli(argv: List[str]) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        main(argv)
    return buf.getvalue()


if __name__ == "__main__":
    raise SystemExit(main())
