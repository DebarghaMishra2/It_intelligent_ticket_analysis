from src.api import database


def test_feedback_is_saved_for_existing_audit_request(client):
    database.create_audit_record(
        request_id="test-request-feedback",
        ticket_id="TEST-FEEDBACK-001",
        endpoint="/api/v1/tickets/recommend",
        input_text="Cannot access Outlook.",
        output={"status": "insufficient_resolution_evidence"},
    )

    response = client.post(
        "/api/v1/feedback",
        json={
            "request_id": "test-request-feedback",
            "ticket_id": "TEST-FEEDBACK-001",
            "feedback": "unresolved",
            "notes": "No approved knowledge article was available.",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "saved"
    assert body["feedback"] == "unresolved"
    assert body["ticket_id"] == "TEST-FEEDBACK-001"


def test_feedback_returns_404_for_unknown_request(client):
    response = client.post(
        "/api/v1/feedback",
        json={
            "request_id": "does-not-exist",
            "feedback": "incorrect",
        },
    )

    assert response.status_code == 404
