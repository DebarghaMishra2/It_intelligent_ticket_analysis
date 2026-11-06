"""Unified Incident Workbench for the IT-Sentinel Streamlit dashboard.

This is a thin UI layer. Classification, retrieval, recommendations,
feedback and audit operations remain in the existing FastAPI backend.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from frontend.api_client import APIError, ITSentinelClient


CATEGORIES = [
    "Active Directory",
    "Computer-Services",
    "EOL",
    "Fileservice",
    "O365",
    "Software",
    "Support general",
]


def _show_error(error: Exception) -> None:
    st.error(str(error))
    if isinstance(error, APIError) and error.status_code:
        st.caption(f"HTTP status: {error.status_code}")


def _save_request(kind: str, result: Dict[str, Any], ticket_id: str) -> None:
    request_id = result.get("request_id")
    if not request_id:
        return
    st.session_state.setdefault("its_requests", {})
    st.session_state["its_requests"][kind] = {
        "request_id": request_id,
        "ticket_id": ticket_id,
    }


def _render_similar_tickets(items: List[Dict[str, Any]]) -> None:
    if not items:
        st.info("No similar historical tickets were returned.")
        return

    rows = []
    for item in items:
        rows.append(
            {
                "Rank": item.get("rank"),
                "Ticket ID": item.get("ticket_id"),
                "Category": item.get("category"),
                "BM25 rank": item.get("bm25_rank"),
                "Semantic rank": item.get("semantic_rank"),
                "RRF score": item.get("rrf_score"),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    for item in items:
        heading = (
            f"#{item.get('rank', '?')} · {item.get('ticket_id', 'Unknown ticket')} · "
            f"{item.get('category', 'Unclassified')}"
        )
        with st.expander(heading):
            st.write(item.get("description") or "No description is available.")
            score_details = {
                key: item.get(key)
                for key in (
                    "bm25_rank",
                    "bm25_score",
                    "semantic_rank",
                    "semantic_score",
                    "rrf_score",
                    "retrieval_score",
                    "retrieval_score_type",
                )
                if item.get(key) is not None
            }
            if score_details:
                st.json(score_details)
            st.caption(
                "Source type: historical ticket description. This is a precedent, "
                "not a verified resolution or proof that the issue has the same cause."
            )


def _render_knowledge(result: Dict[str, Any]) -> None:
    status = result.get("status")
    matches = result.get("approved_knowledge_matches") or []
    recommendation = result.get("recommendation")

    st.markdown("#### Approved knowledge and resolution evidence")
    if status == "insufficient_resolution_evidence" or not recommendation:
        st.warning(
            result.get(
                "message",
                "No approved resolution evidence was found. Human review is required.",
            )
        )
        st.caption(
            "Historical ticket descriptions are not resolution instructions. "
            "Do not infer or execute a fix from these records alone."
        )
        return

    st.success(
        "Approved knowledge candidates were found. An analyst must validate "
        "the article and steps before taking action."
    )
    steps = recommendation.get("steps") or []
    if steps:
        for index, step in enumerate(steps, start=1):
            st.write(f"{index}. {step}")
    st.write(
        "**Source:** "
        f"{recommendation.get('source_name', '—')} · "
        f"**Article:** {recommendation.get('source_article_id', '—')}"
    )
    if recommendation.get("source_reference"):
        st.write(f"**Reference:** {recommendation['source_reference']}")

    if matches:
        with st.expander("Inspect approved knowledge evidence"):
            st.json(matches)


def render_incident_workbench(client: ITSentinelClient) -> None:
    """Render the end-to-end analyst workflow using the existing API endpoints."""

    st.subheader("Incident Workbench")
    st.write(
        "Submit an incident once. IT-Sentinel will classify it, estimate the "
        "demonstration priority, retrieve historical precedents, and check the "
        "approved knowledge base. Review all evidence before acting."
    )

    with st.form("its_incident_workbench_form", clear_on_submit=False):
        top_left, top_right = st.columns([1, 1])
        ticket_id = top_left.text_input(
            "Ticket / incident ID",
            value="INC-DEMO-001",
            max_chars=100,
            help="Use the same ID to group the incident's audit events.",
        )
        top_k = top_right.slider("Similar historical tickets", 1, 10, 5)

        impact_col, urgency_col = st.columns(2)
        impact = impact_col.selectbox(
            "Impact",
            ["Not provided", "low", "medium", "high"],
            index=2,
        )
        urgency = urgency_col.selectbox(
            "Urgency",
            ["Not provided", "low", "medium", "high"],
            index=3,
        )
        description = st.text_area(
            "Incident description",
            height=160,
            placeholder=(
                "Example: Unable to access Outlook after changing my password. "
                "Authentication keeps failing. Include the application, symptoms, "
                "error text and steps already attempted."
            ),
        )
        submitted = st.form_submit_button(
            "Analyze incident and retrieve evidence",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        clean_ticket_id = ticket_id.strip()
        clean_description = description.strip()
        if not clean_ticket_id:
            st.warning("Enter a ticket ID so the analysis can be linked to its audit history.")
        elif len(clean_description) < 5:
            st.warning("Enter an incident description containing at least five characters.")
        else:
            analyze_payload: Dict[str, Any] = {
                "ticket_id": clean_ticket_id,
                "description": clean_description,
            }
            if impact != "Not provided":
                analyze_payload["impact"] = impact
            if urgency != "Not provided":
                analyze_payload["urgency"] = urgency

            recommend_payload = {
                "ticket_id": clean_ticket_id,
                "description": clean_description,
                "top_k": top_k,
            }

            analysis_result = None
            recommendation_result = None
            try:
                with st.spinner("Classifying incident and retrieving supporting evidence..."):
                    analysis_result = client.analyze_ticket(analyze_payload)
                    _save_request("workbench_analysis", analysis_result, clean_ticket_id)
                    recommendation_result = client.recommend(recommend_payload)
                    _save_request(
                        "workbench_recommendation", recommendation_result, clean_ticket_id
                    )
                st.session_state["its_workbench_result"] = {
                    "ticket_id": clean_ticket_id,
                    "description": clean_description,
                    "analysis": analysis_result,
                    "recommendation": recommendation_result,
                }
                st.success("Incident analysis completed. Review the results below.")
            except APIError as error:
                # Preserve a completed classification if the recommendation call fails.
                if analysis_result:
                    st.session_state["its_workbench_result"] = {
                        "ticket_id": clean_ticket_id,
                        "description": clean_description,
                        "analysis": analysis_result,
                        "recommendation": None,
                    }
                _show_error(error)

    bundle = st.session_state.get("its_workbench_result")
    if not bundle:
        st.info(
            "No incident has been analyzed in this session yet. Enter a ticket above "
            "and select **Analyze incident and retrieve evidence**."
        )
        return

    analysis = bundle.get("analysis") or {}
    recommendation = bundle.get("recommendation") or {}
    classification = analysis.get("classification") or {}
    ticket_id = bundle.get("ticket_id", "")

    st.divider()
    st.markdown(f"### Analysis result · `{ticket_id}`")
    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Predicted category", classification.get("predicted_category", "—")
    )
    probability = classification.get("model_probability")
    metric_cols[1].metric(
        "Model probability", f"{float(probability):.1%}" if probability is not None else "—"
    )
    metric_cols[2].metric("Priority estimate", analysis.get("priority") or "Not assigned")
    metric_cols[3].metric("Confidence band", analysis.get("confidence_band", "—"))

    if analysis.get("human_review_recommended"):
        st.warning(
            "Human review recommended. The model probability is below the configured "
            f"review threshold ({analysis.get('review_threshold', 'not supplied')})."
        )
    else:
        st.info(
            "The configured confidence threshold was met. Analyst validation is still "
            "required before any operational action."
        )

    if classification.get("top_categories"):
        with st.expander("View alternative predicted categories"):
            chart_frame = pd.DataFrame(classification["top_categories"])
            if {"category", "probability"}.issubset(chart_frame.columns):
                st.bar_chart(chart_frame.set_index("category")["probability"])

    if analysis.get("priority_note"):
        st.caption(analysis["priority_note"])

    if analysis.get("limitations"):
        with st.expander("Classifier and priority limitations"):
            for note in analysis["limitations"]:
                st.write(f"- {note}")

    st.divider()
    evidence_col, knowledge_col = st.columns([1.25, 1])
    with evidence_col:
        st.markdown("### Similar historical tickets")
        similar = recommendation.get("similar_historical_tickets") or []
        _render_similar_tickets(similar)
    with knowledge_col:
        _render_knowledge(recommendation)
        if recommendation.get("human_review_required"):
            st.warning("Analyst review is required before deciding on remediation.")

    with st.expander("Request IDs and traceability"):
        st.write("Classification request ID:", analysis.get("request_id", "—"))
        st.write("Recommendation request ID:", recommendation.get("request_id", "—"))
        st.caption(
            "Each API call is separately audited. The shared ticket ID groups these "
            "events in the ticket audit endpoint."
        )

    st.divider()
    st.markdown("### Analyst decision")
    recommendation_request_id = recommendation.get("request_id")
    if not recommendation_request_id:
        st.info(
            "A recommendation request ID is not available because the recommendation "
            "API call did not complete. Re-run the incident analysis before submitting feedback."
        )
    else:
        with st.form("its_workbench_feedback_form"):
            decision = st.selectbox(
                "Was the analysis / evidence useful?",
                ["useful", "incorrect", "unresolved"],
                format_func=lambda value: {
                    "useful": "Useful",
                    "incorrect": "Incorrect",
                    "unresolved": "Unresolved — needs further investigation",
                }[value],
            )
            notes = st.text_area(
                "Reviewer notes",
                max_chars=5000,
                height=90,
                placeholder="Explain what was useful, incorrect, or still missing...",
            )
            feedback_submitted = st.form_submit_button(
                "Save analyst decision", type="primary", use_container_width=True
            )
        if feedback_submitted:
            feedback_payload = {
                "request_id": recommendation_request_id,
                "ticket_id": ticket_id,
                "feedback": decision,
                "notes": notes.strip() or None,
            }
            try:
                saved = client.submit_feedback(feedback_payload)
                st.session_state["its_workbench_feedback"] = saved
                st.success("Analyst decision saved to the audit database.")
            except APIError as error:
                _show_error(error)

    if st.session_state.get("its_workbench_feedback"):
        saved = st.session_state["its_workbench_feedback"]
        if saved.get("ticket_id") == ticket_id:
            with st.expander("Last saved analyst decision"):
                st.json(saved)

    st.divider()
    if st.button("Load audit history for this incident", use_container_width=True):
        try:
            audit = client.audit_ticket(ticket_id)
            st.session_state["its_workbench_audit"] = audit
        except APIError as error:
            _show_error(error)

    audit = st.session_state.get("its_workbench_audit")
    if audit and audit.get("ticket_id") == ticket_id:
        st.markdown("### Incident audit trail")
        st.metric("Recorded API events", audit.get("record_count", 0))
        for record in audit.get("records", []):
            with st.expander(
                f"{record.get('endpoint', 'API event')} · {record.get('created_at', '')} · "
                f"{record.get('request_id', '')}"
            ):
                st.write("Analyst decision:", record.get("analyst_decision") or "Not reviewed")
                if record.get("analyst_notes"):
                    st.write("Reviewer notes:", record["analyst_notes"])
                st.write("Input:", record.get("input_text", ""))
                st.json(record.get("output", {}))
