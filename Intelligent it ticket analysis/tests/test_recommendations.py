def test_recommendation_falls_back_when_no_resolution_evidence_exists(
    client, monkeypatch
):
    from src.api.routes import tickets as tickets_routes

    fake_result = {
        "status": "insufficient_resolution_evidence",
        "recommendation": None,
        "approved_knowledge_matches": [],
        "similar_historical_tickets": [],
        "human_review_required": True,
        "message": "No verified resolution evidence is available.",
        "safety_note": "Decision support only; no remediation is executed.",
    }
    monkeypatch.setattr(
        tickets_routes, "build_recommendation", lambda *args, **kwargs: fake_result
    )

    response = client.post(
        "/api/v1/tickets/recommend",
        json={
            "ticket_id": "TEST-REC-001",
            "description": "Outlook authentication fails after password change.",
            "top_k": 5,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "insufficient_resolution_evidence"
    assert body["recommendation"] is None
    assert body["human_review_required"] is True
    assert body["request_id"]
