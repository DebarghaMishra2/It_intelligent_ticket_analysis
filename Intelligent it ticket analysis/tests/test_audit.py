from src.api import database


def test_ticket_audit_lists_saved_requests(client):
    database.create_audit_record(
        request_id="test-audit-001",
        ticket_id="TEST-AUDIT-001",
        endpoint="/api/v1/tickets/analyze",
        input_text="Outlook login not working.",
        output={"predicted_category": "O365"},
    )

    response = client.get("/api/v1/audit/tickets/TEST-AUDIT-001")

    assert response.status_code == 200
    body = response.json()
    assert body["record_count"] == 1
    assert body["records"][0]["request_id"] == "test-audit-001"
    assert body["records"][0]["output"]["predicted_category"] == "O365"


def test_request_audit_includes_feedback_history(client):
    database.create_audit_record(
        request_id="test-audit-feedback-001",
        ticket_id="TEST-AUDIT-FEEDBACK-001",
        endpoint="/api/v1/tickets/recommend",
        input_text="Outlook login not working.",
        output={"status": "insufficient_resolution_evidence"},
    )
    database.add_feedback(
        request_id="test-audit-feedback-001",
        ticket_id=None,
        feedback="unresolved",
        notes="Needs analyst investigation.",
    )

    response = client.get(
        "/api/v1/audit/requests/test-audit-feedback-001"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["analyst_decision"] == "unresolved"
    assert body["analyst_notes"] == "Needs analyst investigation."
    assert len(body["feedback_history"]) == 1
    assert body["feedback_history"][0]["feedback"] == "unresolved"


def test_missing_request_audit_returns_404(client):
    response = client.get("/api/v1/audit/requests/not-a-real-request")
    assert response.status_code == 404
