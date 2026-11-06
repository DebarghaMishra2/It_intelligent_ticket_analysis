def test_analyze_classifies_ticket_and_assigns_demo_priority(client):
    response = client.post(
        "/api/v1/tickets/analyze",
        json={
            "ticket_id": "TEST-CLASS-001",
            "description": "Unable to access Outlook after changing my password.",
            "impact": "medium",
            "urgency": "high",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["ticket_id"] == "TEST-CLASS-001"
    assert body["classification"]["predicted_category"] == "O365"
    assert body["classification"]["model_probability"] == 0.55
    assert body["confidence_band"] == "low"
    assert body["human_review_recommended"] is True
    assert body["priority"] == "P2"
    assert body["request_id"]


def test_analyze_rejects_description_that_is_too_short(client):
    response = client.post(
        "/api/v1/tickets/analyze",
        json={"description": "bad"},
    )

    assert response.status_code == 422
