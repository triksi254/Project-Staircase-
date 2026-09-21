"""Counsellor test platform: chat + live lead scoring + lead identity tracking.

Import-guard: importing this module must NEVER require streamlit, so pytest
collection and headless runs work without the app stack installed. All
streamlit access goes through :func:`_st`, which raises an actionable error
only when the UI entrypoint (:func:`main`) is actually invoked.

Lead lifecycle (in-memory only, nothing persisted to disk):
  * first render with ``st.session_state.lead_id is None`` -> create lead,
    ``is_new=True``, ``turns_since_new=0``;
  * "Close Lead & Start New" archives the current lead (id, final score,
    label, turn count **plus the full chat transcript**) into
    ``st.session_state.leads_archive`` and starts a fresh lead with a
    reset chat + tracker;
  * the "NEW LEAD" badge shows while ``turns_since_new < 3``;
  * ``lead_counter`` increments per session (``LEAD-YYYYMMDD-NNNN``).

Dissertation downloads: every lead (active + each archived one) offers
``st.download_button`` exports of its transcript in JSON (full record),
Markdown (appendix-ready), and CSV (one row per turn) via
:func:`transcript_to_json`, :func:`transcript_to_markdown`,
:func:`transcript_to_csv`. Each turn also records the escalation decision,
priority, lead label/score and the retrieval backend + gate that produced it.

Per-turn pipeline (:func:`_answer_query`): user text -> retriever (Sentence-BERT
when it can start, else TF-IDF; each with its own abstention gate) ->
``responder.respond`` (grounded answer or abstention) ->
``SessionTracker.add_turn`` -> **then** the lead is scored (rule over
chat-observable topics + the provisional engagement ML score, blended at the
configured alpha, mapped by ``score_to_label``) -> escalation is decided from
that post-turn label, so the decision reflects the message just sent.
"""
from __future__ import annotations

import csv
import functools
import io
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

PROJECT_TITLE = "Counsellor Test Platform"
NEW_LEAD_BADGE_TURNS = 3

#: Columns appended to the transcript CSV/rows (blank for older entries).
DECISION_FIELDS = ("escalate", "priority", "lead_label", "lead_score",
                   "retrieval_backend")


def _st():
    try:
        import streamlit as st
    except ImportError as exc:
        raise ImportError(
            "dashboard/app.py requires streamlit: "
            "pip install -r requirements-app.txt") from exc
    return st


def new_lead_id(counter: int, now: Optional[datetime] = None) -> str:
    """LEAD-YYYYMMDD-NNNN badge id for a 1-based per-session counter."""
    ts = now or datetime.now(timezone.utc)
    return f"LEAD-{ts.strftime('%Y%m%d')}-{counter:04d}"


def should_show_new_badge(turns_since_new: int,
                          window: int = NEW_LEAD_BADGE_TURNS) -> bool:
    """Badge is visible for the first ``window`` turns after creation."""
    return int(turns_since_new) < int(window)


def archive_entry(lead_id: str, tracker, rule: float, hybrid: float,
                  label: str, chat: Optional[List[Dict[str, Any]]] = None,
                  turns_log: Optional[List[Dict[str, Any]]] = None,
                  started_at: Optional[str] = None) -> Dict[str, Any]:
    """Snapshot a closing lead for the archive panel (last-5 display).

    For dissertation evidence the full chat transcript is stored alongside
    the final score/label: ``chat`` holds the rendered user/assistant
    messages and ``turns_log`` the per-turn classifier signals (category,
    confidence, timestamp) and routing decision. Both default to ``[]`` so old
    callers (``archive_entry(lead_id, tracker, rule, hybrid, label)``) keep
    working.
    """
    feats = tracker.features() if tracker is not None else {}
    return {
        "lead_id": lead_id,
        "final_rule_score": round(float(rule), 4),
        "final_hybrid_score": round(float(hybrid), 4),
        "final_label": label,
        "turns": int(feats.get("message_count", 0)),
        "closed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "started_at": started_at,
        "chat": list(chat or []),
        "turns_log": list(turns_log or []),
    }


def transcript_rows(entry: Dict[str, Any]) -> List[Dict[str, Any]]:
    """One row per user turn: user msg + assistant reply + signals + decision.

    ``turns_log`` carries the classifier signals (category/confidence/ts) and,
    for turns recorded by :func:`_answer_query`, the routing decision
    (``DECISION_FIELDS``); older entries yield blanks. The rendered assistant
    reply is recovered from ``chat`` (layout ``[user0, asst0, user1, asst1,
    ...]``). Falls back to pairing raw ``chat`` messages when ``turns_log`` is
    empty.
    """
    chat = list(entry.get("chat") or [])
    turns = list(entry.get("turns_log") or [])
    rows: List[Dict[str, Any]] = []
    if turns:
        for i, t in enumerate(turns):
            assistant_text = str(t.get("assistant_text") or "")
            if not assistant_text:
                idx = 2 * i + 1
                if 0 <= idx < len(chat):
                    assistant_text = str(chat[idx].get("text", ""))
            row = {
                "turn_no": i + 1,
                "timestamp": t.get("ts", ""),
                "user_message": t.get("user_msg", ""),
                "assistant_response": assistant_text,
                "category": t.get("category", ""),
                "confidence": t.get("confidence", 0.0),
            }
            for key in DECISION_FIELDS:
                row[key] = t.get(key, "")
            rows.append(row)
        return rows
    # Fallback: pair raw chat messages when no per-turn signals exist.
    for i in range(0, len(chat), 2):
        user_text = str(chat[i].get("text", "")) if i < len(chat) else ""
        asst_text = str(chat[i + 1].get("text", "")) if i + 1 < len(chat) else ""
        row = {
            "turn_no": len(rows) + 1,
            "timestamp": "",
            "user_message": user_text,
            "assistant_response": asst_text,
            "category": "",
            "confidence": 0.0,
        }
        for key in DECISION_FIELDS:
            row[key] = ""
        rows.append(row)
    return rows


def transcript_to_json(entry: Dict[str, Any]) -> str:
    """Full lead record (scores + transcript) as pretty JSON (UTF-8)."""
    return json.dumps(entry, indent=2, ensure_ascii=False)


def transcript_to_markdown(entry: Dict[str, Any]) -> str:
    """Human-readable transcript for dissertation appendices."""
    lines = [
        f"# Chat transcript — {entry.get('lead_id', 'LEAD-unknown')}",
        "",
        f"- Final label: **{entry.get('final_label', '')}**",
        f"- Final hybrid score: **{entry.get('final_hybrid_score', '')}**",
        f"- Final rule score: **{entry.get('final_rule_score', '')}**",
        f"- Turns: **{entry.get('turns', len(transcript_rows(entry)))}**",
        f"- Started: `{entry.get('started_at', '')}`",
        f"- Closed/exported: `{entry.get('closed_at') or entry.get('exported_at', '')}`",
        "",
        "## Conversation",
        "",
    ]
    rows = transcript_rows(entry)
    if not rows:
        lines.append("_No turns yet — the lead has no messages._")
    for r in rows:
        meta = f"Turn {r['turn_no']}"
        if r.get("category"):
            try:
                conf = float(r.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
            meta += f" · {r['category']} (confidence {conf:.2f})"
        if r.get("timestamp"):
            meta += f" · {r['timestamp']}"
        lines += [f"### {meta}", "", f"**User:** {r['user_message']}",
                  "", f"**Assistant:** {r['assistant_response']}", ""]
        decision = _decision_line(r)
        if decision:
            lines += [decision, ""]
    return "\n".join(lines)


def _decision_line(row: Dict[str, Any]) -> str:
    """One markdown line describing the routing decision (empty if unknown)."""
    if row.get("lead_label") in ("", None) and row.get("escalate") in ("", None):
        return ""
    parts = []
    if row.get("lead_label") not in ("", None):
        try:
            parts.append(f"Lead: **{row['lead_label']}** "
                         f"({float(row.get('lead_score')):.2f})")
        except (TypeError, ValueError):
            parts.append(f"Lead: **{row['lead_label']}**")
    if row.get("escalate") not in ("", None):
        parts.append("Escalated to counsellor "
                     f"(priority {row.get('priority') or 'n/a'})"
                     if row["escalate"] else "Not escalated")
    if row.get("retrieval_backend"):
        parts.append(f"retrieval: {row['retrieval_backend']}")
    return "_" + " · ".join(parts) + "_"


def transcript_to_csv(entry: Dict[str, Any]) -> str:
    """One row per turn (turn_no, timestamps, messages, signals, decision)."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=[
        "turn_no", "timestamp", "user_message", "assistant_response",
        "category", "confidence", *DECISION_FIELDS])
    writer.writeheader()
    for r in transcript_rows(entry):
        writer.writerow(r)
    return buf.getvalue()


def build_live_entry(st, tracker=None, scored: Optional[Dict[str, Any]] = None
                     ) -> Dict[str, Any]:
    """Snapshot the *active* lead (for in-progress download buttons)."""
    if tracker is None:
        tracker = _get_tracker(st)
    if scored is None:
        scored = _score_turn(tracker)
    info = st.session_state.get("tracker_state") or {}
    entry = archive_entry(
        st.session_state.get("lead_id", "LEAD-unknown"), tracker,
        scored["rule"], scored["hybrid"], scored["label"],
        chat=st.session_state.get("chat", []),
        turns_log=st.session_state.get("turns_log", []),
        started_at=info.get("started_at"),
    )
    # Not closed yet — mark the export timestamp instead.
    entry["exported_at"] = entry.pop("closed_at")
    return entry


# --------------------------------------------------------------------------- #
# retrieval backend + gate
# --------------------------------------------------------------------------- #
def _get_retriever():
    """Shared retriever: Sentence-BERT when it can actually start, else TF-IDF.

    The SBERT model is loaded *here* (not lazily on the first query) so a
    missing model/offline cache falls back cleanly instead of crashing a chat
    turn. Any failure is logged with its reason; the backend in use is exposed
    via :func:`_backend_of` and shown in the UI. (An earlier version imported
    ``load_corpus`` from ``chatbot.embeddings`` -- it lives in
    ``chatbot.retriever`` -- and the ``ImportError`` was swallowed, so the
    dashboard silently ran TF-IDF every time.)
    """
    import logging
    log = logging.getLogger(__name__)
    from chatbot.retriever import Retriever, load_corpus
    entries = load_corpus()
    try:
        from chatbot.embeddings import SbertRetriever, available
        if available():
            retriever = SbertRetriever(entries)
            # Force the model load inside the try. Assigned, not a bare
            # expression: Streamlit "magic" renders any bare expression in this
            # file, which printed the whole SentenceTransformer repr onto the page.
            _ = retriever.model
            log.info("dashboard: using SBERT retriever (model=%s, corpus=%d)",
                     retriever.model_name, len(retriever.entries))
            return retriever
        log.warning("dashboard: sentence-transformers not installed — "
                    "using TF-IDF (pip install -r requirements-ml.txt)")
    except Exception as exc:  # ImportError, offline/missing HF cache, ...
        log.warning("dashboard: SBERT unavailable (%s: %s) — using TF-IDF",
                    type(exc).__name__, exc)
    log.info("dashboard: using TF-IDF retriever (corpus=%d entries)", len(entries))
    return Retriever(entries)


def _cached_retriever(st):
    """``_get_retriever`` cached across Streamlit reruns (model load is slow)."""
    cache = getattr(st, "cache_resource", None)
    if cache is None:
        return _get_retriever()
    return cache(show_spinner="Loading retrieval model…")(_get_retriever)()


def _backend_of(retriever) -> str:
    """``'sbert'`` or ``'tfidf'`` (unknown retrievers count as TF-IDF)."""
    return str(getattr(retriever, "backend", "tfidf"))


def _gate_for(backend: str) -> float:
    """Abstention gate for a backend.

    SBERT uses the evaluated operating point (``EVALUATED_GATE``, on the raw
    cosine scale it was studied on). TF-IDF scores sit on a different scale
    (cosine + keyword bonus): at 0.60 it withheld 90-100% of answerable gold
    queries, so the TF-IDF fallback uses the permissive default instead.
    """
    from chatbot.responder import DEFAULT_MIN_CONFIDENCE, EVALUATED_GATE
    return EVALUATED_GATE if backend == "sbert" else DEFAULT_MIN_CONFIDENCE


# --------------------------------------------------------------------------- #
# lead scoring
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=1)
def _default_alpha() -> float:
    """Blend weight from ``artifacts/config.json`` (else ``DEFAULT_ALPHA``)."""
    from leads.hybrid import DEFAULT_ALPHA, get_default_alpha, load_config_json
    cfg = load_config_json()
    return float(get_default_alpha({"config": cfg})) if cfg else DEFAULT_ALPHA


def _score_turn(tracker, alpha: Optional[float] = None) -> Dict[str, Any]:
    """Score the current session: rule, ML (if model loaded), hybrid, label.

    A tracker with no turns is an empty lead: skip the ML prediction
    entirely (no features to score) and return a neutral 0.00 state with
    ``ml_proba=None`` so the panel renders its awaiting placeholders.
    """
    import logging
    from leads.hybrid import score_to_label, predict_ml_proba

    alpha = _default_alpha() if alpha is None else float(alpha)
    if len(tracker) == 0:
        logging.getLogger(__name__).info(
            "dashboard turn: no turns yet — awaiting first message")
        return {"rule": 0.0, "ml_proba": None, "hybrid": 0.0,
                "label": "Awaiting first message", "alpha": alpha}

    rule = tracker.rule_score()
    ml_proba = predict_ml_proba(
        tracker.rubric_evidence(), engagement=tracker.features())

    logger = logging.getLogger(__name__)
    hybrid = _hybrid_from_proba(rule, ml_proba, alpha)
    if ml_proba is not None:
        logger.info("dashboard turn: ml_proba=%s rule=%.4f hybrid=%.4f",
                    [round(p, 4) for p in ml_proba], rule, hybrid)
    else:
        logger.info("dashboard turn: ml_proba=None rule=%.4f (model not loaded)",
                    rule)
    return {"rule": rule, "ml_proba": ml_proba, "hybrid": hybrid,
            "label": score_to_label(hybrid), "alpha": alpha}


def _hybrid_from_proba(rule: float, ml_proba: Optional[List[float]],
                        alpha: float) -> float:
    """Hybrid score from rule + the model's 3-class ``predict_proba`` output.

    ``ml_proba`` (``[P(Cold), P(Warm), P(Hot)]``) is forwarded to
    ``leads.hybrid.hybrid_score`` as its ``ml_score`` argument, which maps the
    3-element array to a scalar via expected value before blending. Passing
    ``None`` (no model loaded) keeps the score rule-only.
    """
    from leads.hybrid import hybrid_score
    return float(hybrid_score(rule, ml_proba, alpha=alpha))


# The multi-intent guard lives in ``chatbot.intent_guard`` so ``respond()`` can
# report which groups matched; the old private names stay as aliases.
from chatbot.intent_guard import INTENT_GROUPS as _INTENT_GROUPS  # noqa: E402
from chatbot.intent_guard import NOT_INTENT as _NOT_INTENT  # noqa: E402
from chatbot.intent_guard import is_multi_intent as _is_multi_intent  # noqa: E402
from chatbot.intent_guard import kw_in as _kw_in  # noqa: E402,F401


def _ensure_lead(st) -> None:
    """Create the session's first lead lazily (is_new, counter, tracker)."""
    st.session_state.setdefault("lead_id", None)
    st.session_state.setdefault("lead_counter", 0)
    st.session_state.setdefault("turns_since_new", 0)
    st.session_state.setdefault("is_new", False)
    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("turns_log", [])
    st.session_state.setdefault("leads_archive", [])
    st.session_state.setdefault("tracker_state", None)
    if st.session_state.lead_id is None:
        st.session_state.lead_counter += 1
        st.session_state.lead_id = new_lead_id(st.session_state.lead_counter)
        st.session_state.is_new = True
        st.session_state.turns_since_new = 0
        st.session_state.tracker_state = {
            "lead_id": st.session_state.lead_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }


def _get_tracker(st):
    from chatbot.session_features import SessionTracker
    info = st.session_state.tracker_state or {}
    started = (datetime.fromisoformat(info["started_at"])
               if info.get("started_at") else datetime.now(timezone.utc))
    tracker = SessionTracker(
        info.get("lead_id", st.session_state.lead_id or "LEAD-unknown"),
        started)
    for turn in st.session_state.get("turns_log", []):
        tracker.add_turn(turn["user_msg"],
                         {"category": turn["category"],
                          "confidence": turn["confidence"]},
                         datetime.fromisoformat(turn["ts"]))
    return tracker


def _close_lead_and_start_new(st) -> None:
    """Archive current lead, then reset chat + tracker with a new id."""
    tracker = _get_tracker(st)
    scored = _score_turn(tracker)
    if st.session_state.lead_id is not None:
        info = st.session_state.get("tracker_state") or {}
        st.session_state.leads_archive.append(archive_entry(
            st.session_state.lead_id, tracker, scored["rule"],
            scored["hybrid"], scored["label"],
            chat=st.session_state.get("chat", []),
            turns_log=st.session_state.get("turns_log", []),
            started_at=info.get("started_at")))
    st.session_state.lead_counter += 1
    st.session_state.lead_id = new_lead_id(st.session_state.lead_counter)
    st.session_state.is_new = True
    st.session_state.turns_since_new = 0
    st.session_state.chat = []
    st.session_state.turns_log = []
    st.session_state.tracker_state = {
        "lead_id": st.session_state.lead_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


def _answer_query(query: str, tracker, retriever,
                  gate: Optional[float] = None) -> Dict[str, Any]:
    """One chat turn: respond, record it, then score and route the lead.

    The lead is scored **after** the turn is recorded, and the escalation
    decision is taken from that post-turn label, so a message that turns a
    lead Hot escalates immediately (it used to use the pre-turn score).
    A multi-intent query abstains via ``force_abstain`` on the *real* query.
    """
    from chatbot.responder import EscalationPolicy, decide_escalation, respond
    backend = _backend_of(retriever)
    gate = _gate_for(backend) if gate is None else float(gate)

    # rule_score is a placeholder: the lead fields are recomputed below from
    # the post-turn score, so respond() is only asked for answer/abstain.
    out = respond(query, retriever, rule_score=0.0, ml_score=None,
                  policy=EscalationPolicy(alpha=1.0, min_confidence=gate),
                  top_k=3, force_abstain=_is_multi_intent(query))
    tracker.add_turn(query, out, datetime.now(timezone.utc))

    scored = _score_turn(tracker)
    escalate, priority = decide_escalation(scored["label"], out["abstained"])
    out.update(lead_score=scored["hybrid"], lead_label=scored["label"],
               rule_score=scored["rule"], ml_proba=scored["ml_proba"],
               escalate=escalate, priority=priority,
               retrieval_backend=backend, gate=gate)
    return out


class _WatcherNoiseFilter(logging.Filter):
    """Drop Streamlit's file-watcher probe errors for transformers / torch.

    Once ``transformers`` is imported (SBERT), the watcher calls
    ``hasattr(module, "__path__")`` on every entry of ``sys.modules``. Many of
    transformers' lazy submodules import ``torchvision`` (not installed) and
    raise ``ModuleNotFoundError`` instead of ``AttributeError``, so Streamlit
    logged one ~20-line traceback per module (~400 of them, ~9,000 lines). The
    errors are harmless; errors about any *other* module still pass through.
    """

    _PREFIXES = ("Examining the path of transformers.",
                 "Examining the path of torch.")

    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(self._PREFIXES)


def _quiet_watcher_noise() -> None:
    """Attach :class:`_WatcherNoiseFilter` to the watcher's logger (once)."""
    lg = logging.getLogger("streamlit.watcher.local_sources_watcher")
    if not any(isinstance(f, _WatcherNoiseFilter) for f in lg.filters):
        lg.addFilter(_WatcherNoiseFilter())


def _category_chart(st, counts: Dict[str, int]) -> None:
    """Turns per category as a static horizontal bar chart.

    ``st.bar_chart`` attaches zoom/pan *scale bindings*, which Vega-Lite rejects
    on a categorical axis ("Scale bindings are currently only supported for
    scales with unbinned, continuous domains") -- one browser-console warning
    per render. A non-interactive Altair bar has no bindings and no warning.
    """
    try:
        import altair as alt
        import pandas as pd
    except ImportError:
        st.write(dict(counts))
        return
    df = pd.DataFrame({"category": list(counts), "turns": list(counts.values())})
    chart = (alt.Chart(df).mark_bar()
             .encode(x=alt.X("turns:Q", title="turns",
                             axis=alt.Axis(tickMinStep=1)),
                     y=alt.Y("category:N", sort="-x", title=None),
                     tooltip=["category", "turns"])
             # numeric height: a discrete height ({"step": n}) clashes with
             # Streamlit's fit-y autosize ("Dropping fit-y ... discrete height")
             .properties(width="container", height=max(90, 30 * len(counts))))
    st.altair_chart(chart)


def _closed_or_exported_at(entry: Dict[str, Any]) -> str:
    """Timestamp label that works for archived and in-progress entries."""
    return str(entry.get("closed_at") or entry.get("exported_at") or "")


def _download_row(st, entry: Dict[str, Any], key_prefix: str) -> None:
    """JSON / Markdown / CSV download buttons for one lead's transcript."""
    lead_id = str(entry.get("lead_id", "lead"))
    col_j, col_m, col_c = st.columns(3)
    with col_j:
        st.download_button(
            "⬇️ JSON", transcript_to_json(entry),
            file_name=f"{lead_id}_chat.json", mime="application/json",
            key=f"{key_prefix}_{lead_id}_json")
    with col_m:
        st.download_button(
            "⬇️ Markdown", transcript_to_markdown(entry),
            file_name=f"{lead_id}_chat.md", mime="text/markdown",
            key=f"{key_prefix}_{lead_id}_md")
    with col_c:
        st.download_button(
            "⬇️ CSV", transcript_to_csv(entry),
            file_name=f"{lead_id}_chat.csv", mime="text/csv",
            key=f"{key_prefix}_{lead_id}_csv")


def main() -> None:
    """Streamlit entrypoint (only call site that needs streamlit)."""
    st = _st()

    _quiet_watcher_noise()
    st.set_page_config(page_title=PROJECT_TITLE, layout="wide")
    st.title(PROJECT_TITLE)
    st.caption("Provisional live scoring: a rule score over chat-observable "
               "topics blended with an engagement-based ML score (trained on "
               "synthetic sessions), plus counsellor escalation. "
               "Nothing is persisted to disk.")
    _ensure_lead(st)
    retriever = _cached_retriever(st)
    backend = _backend_of(retriever)
    gate = _gate_for(backend)
    st.caption(f"Retrieval: {backend} · abstains below {gate:.2f}")
    if backend != "sbert":
        st.warning("Sentence-BERT could not start, so retrieval falls back to "
                   "TF-IDF (weaker; see the evaluation). "
                   "Install requirements-ml.txt for the evaluated retriever.")

    left, right = st.columns([3, 2])

    with left:
        st.subheader("Chat")
        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["text"])
        prompt = st.chat_input("Ask about studying in the UK…")
        if prompt:
            tracker = _get_tracker(st)
            with st.chat_message("user"):
                st.markdown(prompt)
            out = _answer_query(prompt, tracker, retriever, gate=gate)
            st.session_state.turns_log.append({
                "user_msg": prompt, "category": out.get("category"),
                "confidence": float(out.get("confidence", 0.0) or 0.0),
                # Keep the plain assistant answer (no display footer) so the
                # dissertation transcript export stays clean even if the
                # on-screen rendering format changes.
                "assistant_text": str(out.get("answer", "")),
                "ts": datetime.now(timezone.utc).isoformat(),
                # routing decision, so exports show what the system did
                "escalate": bool(out["escalate"]),
                "priority": out["priority"],
                "lead_label": out["lead_label"],
                "lead_score": round(float(out["lead_score"]), 4),
                "retrieval_backend": out["retrieval_backend"],
            })
            st.session_state.chat += [
                {"role": "user", "text": prompt},
                {"role": "assistant",
                 "text": f"{out['answer']}\n\n_{out.get('category','')} · "
                         f"confidence {out.get('confidence', 0.0):.2f} · "
                         f"lead {out['lead_label']} ({out['lead_score']:.2f})"
                         f"{' · escalated' if out['escalate'] else ''}_"},
            ]
            st.session_state.turns_since_new += 1
            st.rerun()
        if st.button("🔴 Close Lead & Start New"):
            _close_lead_and_start_new(st)
            st.rerun()

    with right:
        st.subheader("Lead")
        st.markdown(f"## `{st.session_state.lead_id}`")
        if st.session_state.is_new and should_show_new_badge(
                st.session_state.turns_since_new):
            st.success("🆕 NEW LEAD")
        tracker = _get_tracker(st)
        scored = _score_turn(tracker)
        resp = tracker.rubric_evidence()
        flags = resp.get("session_flags", {})
        counts = tracker.category_counts()
        is_empty = len(tracker) == 0
        st.metric("Lead score", "0.000" if is_empty else f"{scored['hybrid']:.3f}",
                  "Awaiting first message" if is_empty else scored["label"])
        if is_empty:
            st.write("ML: awaiting input")
        elif scored.get("ml_proba") is not None:
            ml = scored["ml_proba"]
            st.write(f"Rule score: **{scored['rule']:.3f}** · "
                     f"ML: **{ml[2]:.3f}** (Hot) · "
                     f"{ml[1]:.3f} (Warm) · {ml[0]:.3f} (Cold) · "
                     f"alpha: **{scored['alpha']:.2f}**")
            from leads.hybrid import active_model_info
            info = active_model_info()
            st.caption(f"ML model: {info.get('file')} — "
                       f"{info.get('description') or 'engagement-based, provisional'}")
        else:
            st.write(f"Rule score: **{scored['rule']:.3f}** · "
                     f"ML: **model not loaded** · alpha: **{scored['alpha']:.2f}**")
        from chatbot.responder import EscalationPolicy
        hot_at = EscalationPolicy().hot_threshold
        st.write(f"Escalation: on retrieval confidence < {gate:.2f} OR Hot lead "
                 f"(score >= {hot_at:.2f})")
        st.write("Visa intent: "
                 f"**{'yes' if flags.get('visa_intent_mentioned') else 'no'}**"
                 " · Funding interest: "
                 f"**{'yes' if flags.get('funding_method_present') else 'no'}**")
        if counts:
            _category_chart(st, counts)
        else:
            st.write("No turns yet — the category distribution appears "
                     "after the first message.")
        from chatbot.session_features import live_rule_breakdown
        if not is_empty:
            bd = live_rule_breakdown(resp)
            st.write("Rule score = sum of contributions ÷ observable weight "
                     "(chat-observable components only):")
            for c in sorted(bd["lines"], key=lambda c: -c["contribution"]):
                detail = f" ({c['detail']})" if c.get("detail") else ""
                st.write(f"- {c['feature']}: {c['contribution']:+.3f} "
                         f"of {c['weight']:.3f}{detail}")
            st.write(f"**= {bd['raw']:.3f} ÷ {bd['mass']:.3f} = "
                     f"{bd['score']:.3f}**")
        st.divider()
        st.subheader("Download this chat")
        st.caption("Dissertation evidence: transcript + scores + routing "
                   "decision for the active lead (updates after every message).")
        live_entry = build_live_entry(st, tracker=tracker, scored=scored)
        _download_row(st, live_entry, key_prefix="live")

    st.divider()
    st.subheader("Recently closed leads (last 5)")
    archive: List[Dict[str, Any]] = st.session_state.get("leads_archive", [])
    if not archive:
        st.write("No closed leads yet this session.")
    else:
        st.caption("Dissertation evidence: download each closed lead's full "
                   "chat transcript (JSON / Markdown / CSV).")
        for entry in reversed(archive[-5:]):
            with st.expander(
                    f"`{entry['lead_id']}` — {entry.get('final_label')} "
                    f"({float(entry.get('final_hybrid_score', 0.0)):.3f}), "
                    f"{entry.get('turns')} turns, closed "
                    f"{_closed_or_exported_at(entry)}"):
                rows = transcript_rows(entry)
                if rows:
                    st.write(f"Turn 1 user: {rows[0]['user_message'][:160]}")
                _download_row(st, entry, key_prefix="archived")


if __name__ == "__main__":
    main()
