"""Counsellor test platform: chat + live lead scoring + lead identity tracking.

Import-guard: importing this module must NEVER require streamlit, so pytest
collection and headless runs work without the app stack installed. All
streamlit access goes through :func:`_st`, which raises an actionable error
only when the UI entrypoint (:func:`main`) is actually invoked.

Lead lifecycle (in-memory only, nothing persisted to disk):
  * first render with ``st.session_state.lead_id is None`` -> create lead,
    ``is_new=True``, ``turns_since_new=0``;
  * "New Lead" button archives the current lead (id, final score, label,
    turn count) into ``st.session_state.leads_archive`` and starts a fresh
    lead with a reset chat + tracker;
  * the "NEW LEAD" badge shows while ``turns_since_new < 3``;
  * ``lead_counter`` increments per session (``LEAD-YYYYMMDD-NNNN``).

Per-turn pipeline: user text -> shared TF-IDF ``Retriever`` ->
``chatbot.session_features.SessionTracker.add_turn`` ->
rule score (``leads.rubric``) -> hybrid rule-only (``ml_proba=None``) ->
label via ``score_to_label`` -> ``responder.respond`` for the grounded
answer + escalation flag.
"""
from __future__ import annotations

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
                  label: str) -> Dict[str, Any]:
    """Snapshot a closing lead for the archive panel (last-5 display)."""
    feats = tracker.features() if tracker is not None else {}
    return {
        "lead_id": lead_id,
        "final_rule_score": round(float(rule), 4),
        "final_hybrid_score": round(float(hybrid), 4),
        "final_label": label,
        "turns": int(feats.get("message_count", 0)),
        "closed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _get_retriever():
    from chatbot.retriever import Retriever, load_corpus
    return Retriever(load_corpus())


def _score_turn(tracker, alpha: float = 0.5) -> Dict[str, Any]:
    from leads.hybrid import score_to_label
    rule = tracker.rule_score()
    hybrid = tracker.hybrid(alpha=alpha)
    return {"rule": rule, "hybrid": hybrid, "ml": None,
            "label": score_to_label(hybrid), "alpha": float(alpha)}


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
        st.session_state.leads_archive.append(archive_entry(
            st.session_state.lead_id, tracker, scored["rule"],
            scored["hybrid"], scored["label"]))
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
    # Rule-only hybrid pre-combined so respond() does not re-blend; the
    # escalation label still comes from score_to_label on the same value.
    pre = tracker.hybrid()
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
        st.metric("Lead score", f"{scored['hybrid']:.3f}", scored["label"])
        st.write(f"Rule score: **{scored['rule']:.3f}** · "
                 f"ML score: **rule-only** · alpha: **{scored['alpha']:.2f}**")
        from chatbot.responder import EscalationPolicy
        hot_at = EscalationPolicy().hot_threshold
        esc = ("escalate (Hot)" if scored["label"] == "Hot"
               else "escalate only on low confidence")
        st.write(f"Escalation: **{esc}** (label={scored['label']}, "
                 f"hot≥{hot_at:.2f})")
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
        contribs = score_row(
            {k: v for k, v in resp.items() if k != "session_flags"}
        ).contributions
        top = sorted(contribs, key=lambda c: -abs(c["contribution"]))[:5]
        st.write("Top rubric contributions:")
        for c in top:
            st.write(f"- {c['feature']}: {c['contribution']:+.3f} "
                     f"({c.get('detail', '')})")

    st.divider()
    st.subheader("Recently closed leads (last 5)")
    archive: List[Dict[str, Any]] = st.session_state.get("leads_archive", [])
    if not archive:
        st.write("No closed leads yet this session.")
    else:
        for entry in reversed(archive[-5:]):
            st.write(f"- `{entry['lead_id']}` — {entry['final_label']} "
                     f"({entry['final_hybrid_score']:.3f}), "
                     f"{entry['turns']} turns, closed {entry['closed_at']}")


if __name__ == "__main__":
    main()
