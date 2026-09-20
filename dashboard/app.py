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
:func:`transcript_to_csv`.

Per-turn pipeline: user text -> shared TF-IDF ``Retriever`` ->
``chatbot.session_features.SessionTracker.add_turn`` ->
rule score (``leads.rubric``) -> hybrid rule-only (``ml_proba=None``) ->
label via ``score_to_label`` -> ``responder.respond`` for the grounded
answer + escalation flag.
"""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

PROJECT_TITLE = "Counsellor Test Platform"
NEW_LEAD_BADGE_TURNS = 3


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
    confidence, timestamp). Both default to ``[]`` so old callers
    (``archive_entry(lead_id, tracker, rule, hybrid, label)``) keep working.
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
    """One row per user turn: user msg + assistant reply + signals.

    ``turns_log`` carries the classifier signals (category/confidence/ts)
    per user turn; the rendered assistant reply is recovered from ``chat``
    (layout ``[user0, asst0, user1, asst1, ...]``). Falls back to pairing
    raw ``chat`` messages when ``turns_log`` is empty.
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
            rows.append({
                "turn_no": i + 1,
                "timestamp": t.get("ts", ""),
                "user_message": t.get("user_msg", ""),
                "assistant_response": assistant_text,
                "category": t.get("category", ""),
                "confidence": t.get("confidence", 0.0),
            })
        return rows
    # Fallback: pair raw chat messages when no per-turn signals exist.
    for i in range(0, len(chat), 2):
        user_text = str(chat[i].get("text", "")) if i < len(chat) else ""
        asst_text = str(chat[i + 1].get("text", "")) if i + 1 < len(chat) else ""
        rows.append({
            "turn_no": len(rows) + 1,
            "timestamp": "",
            "user_message": user_text,
            "assistant_response": asst_text,
            "category": "",
            "confidence": 0.0,
        })
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
    return "\n".join(lines)


def transcript_to_csv(entry: Dict[str, Any]) -> str:
    """One row per turn (turn_no, timestamps, messages, signals)."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=[
        "turn_no", "timestamp", "user_message", "assistant_response",
        "category", "confidence"])
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


def _get_retriever():
    """Shared retriever for the session. Prefer Sentence-BERT when available.

    When ``sentence-transformers`` is installed the semantic ``SbertRetriever``
    is used (cosine of L2-normalised embeddings); otherwise falls back to the
    TF-IDF ``Retriever``. The selected backend is logged once at import time
    so it is visible in the console / Cloud Logs without a user query.
    """
    try:
        from chatbot.embeddings import SbertRetriever, load_corpus
        from chatbot.embeddings import available as sbert_available
    except ImportError:
        sbert_available = False
        SbertRetriever = None
        load_corpus = None

    if sbert_available and SbertRetriever is not None:
        try:
            entries = load_corpus()
            retriever = SbertRetriever(entries)
            import logging
            logging.getLogger(__name__).info(
                "dashboard: using SBERT retriever (model=%s, corpus=%d entries)",
                retriever.model_name, len(retriever.entries))
            return retriever
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "dashboard: SBERT init failed (%s) — falling back to TF-IDF", exc)

    from chatbot.retriever import Retriever, load_corpus as _load_corpus
    entries = _load_corpus()
    import logging
    logging.getLogger(__name__).info(
        "dashboard: using TF-IDF retriever (corpus=%d entries)", len(entries))
    return Retriever(entries)


def _score_turn(tracker, alpha: float = 0.5) -> Dict[str, Any]:
    """Score the current session: rule, ML (if model loaded), hybrid, label.

    A tracker with no turns is an empty lead: skip the ML prediction
    entirely (no features to score) and return a neutral 0.00 state with
    ``ml_proba=None`` so the panel renders its awaiting placeholders.
    """
    import logging
    from leads.hybrid import score_to_label, predict_ml_proba

    if len(tracker) == 0:
        logging.getLogger(__name__).info(
            "dashboard turn: no turns yet — awaiting first message")
        return {"rule": 0.0, "ml_proba": None, "hybrid": 0.0,
                "label": "Awaiting first message", "alpha": float(alpha)}

    rule = tracker.rule_score()
    ml_proba = predict_ml_proba(
        tracker.rubric_evidence(), engagement=tracker.features())

    logger = logging.getLogger(__name__)
    if ml_proba is not None:
        logger.info(
            "dashboard turn: ml_proba=%s rule=%.4f hybrid=%.4f",
            [round(p, 4) for p in ml_proba], rule,
            _hybrid_from_proba(rule, ml_proba, alpha))
    else:
        logger.info(
            "dashboard turn: ml_proba=None rule=%.4f (model not loaded)",
            rule)

    hybrid = _hybrid_from_proba(rule, ml_proba, alpha)
    return {"rule": rule, "ml_proba": ml_proba, "hybrid": hybrid,
            "label": score_to_label(hybrid), "alpha": float(alpha)}


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


#: Keyword groups for the multi-intent guard: >2 groups hit -> escalate
#: instead of answering (compound queries need a counsellor, not top-1).
_INTENT_GROUPS = (
    ("passport",),
    ("ielts", "english test", "english-test"),
    ("kcse", "grade", "mean grade", "b+", "c+", "gpa"),
    ("data science", "nursing", "business", "course", "degree",
     "masters", "programme", "program", "subject"),
    ("intake", "september", "january", "deadline", "when can i start",
     "when do i apply"),
    ("aston", "bcu", "usw", "rgu", "herts", "salford", "uclan",
     "destination", "which university", "where should i study"),
)

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


def _is_multi_intent(query: str) -> bool:
    q = query.lower()
    hits = sum(any(k in q for k in g) for g in _INTENT_GROUPS)
    return hits > 2


def _answer_query(query: str, tracker, retriever) -> Dict[str, Any]:
    """One chat turn: respond, then record it on the tracker."""
    from chatbot.responder import EscalationPolicy, respond
    scored = _score_turn(tracker)
    ml_proba = scored.get("ml_proba")
    pre = tracker.hybrid(ml_proba=ml_proba)
    if _is_multi_intent(query):
        out = respond("purple monkey dishwasher", retriever,
                      rule_score=pre, ml_score=None,
                      policy=EscalationPolicy(alpha=1.0,
                                              min_confidence=0.60), top_k=3)
        out["query"] = query
        out["category"] = "General Enquiries"
    else:
        out = respond(query, retriever, rule_score=pre, ml_score=None,
                      policy=EscalationPolicy(alpha=1.0,
                                              min_confidence=0.60), top_k=3)
    tracker.add_turn(query, out, datetime.now(timezone.utc))
    return out


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

    st.set_page_config(page_title=PROJECT_TITLE, layout="wide")
    st.title(PROJECT_TITLE)
    st.caption("Rule-only live scoring (provisional) + counsellor "
               "escalation. Nothing is persisted to disk.")
    _ensure_lead(st)
    retriever = _get_retriever()

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
            out = _answer_query(prompt, tracker, retriever)
            scored = _score_turn(tracker)
            st.session_state.turns_log.append({
                "user_msg": prompt, "category": out.get("category"),
                "confidence": float(out.get("confidence", 0.0) or 0.0),
                # Keep the plain assistant answer (no display footer) so the
                # dissertation transcript export stays clean even if the
                # on-screen rendering format changes.
                "assistant_text": str(out.get("answer", "")),
                "ts": datetime.now(timezone.utc).isoformat(),
            })
            st.session_state.chat += [
                {"role": "user", "text": prompt},
                {"role": "assistant",
                 "text": f"{out['answer']}\n\n_{out.get('category','')} · "
                         f"confidence {out.get('confidence', 0.0):.2f} · "
                         f"lead {scored['label']} ({scored['hybrid']:.2f})_"},
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
        else:
            st.write(f"Rule score: **{scored['rule']:.3f}** · "
                     f"ML: **model not loaded** · alpha: **{scored['alpha']:.2f}**")
        from chatbot.responder import EscalationPolicy
        hot_at = EscalationPolicy().hot_threshold
        st.write("Escalation: on low retrieval confidence OR Hot lead "
                 f"(score >= {hot_at:.2f})")
        st.write("Visa intent: "
                 f"**{'yes' if flags.get('visa_intent_mentioned') else 'no'}**"
                 " · Funding interest: "
                 f"**{'yes' if flags.get('funding_method_present') else 'no'}**")
        if counts:
            try:
                import pandas as pd
                st.bar_chart(pd.DataFrame(
                    {"turns": list(counts.values())},
                    index=list(counts.keys())))
            except ImportError:
                st.write(dict(counts))
        else:
            st.write("No turns yet — the category distribution appears "
                     "after the first message.")
        from leads.rubric import score_row
        if not is_empty:
            contribs = score_row(
                {k: v for k, v in resp.items() if k != "session_flags"}
            ).contributions
            top = sorted(contribs, key=lambda c: -abs(c["contribution"]))[:5]
            st.write("Top rubric contributions:")
            for c in top:
                st.write(f"- {c['feature']}: {c['contribution']:+.3f} "
                         f"({c.get('detail', '')})")
        st.divider()
        st.subheader("Download this chat")
        st.caption("Dissertation evidence: transcript + scores for the "
                   "active lead (updates after every message).")
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
