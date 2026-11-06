def test_search_returns_similar_historical_tickets(client, monkeypatch):
    from src.api.routes import tickets as tickets_routes

    fake_results = [
        {
            "rank": 1,
            "ticket_id": "ITS-TRAIN-TEST-001",
            "category": "O365",
            "description": "Outlook authentication issue",
            "bm25_rank": 1,
            "bm25_score": 4.2,
            "semantic_rank": 2,
            "semantic_score": 0.71,
            "rrf_score": 0.032,
            "evidence_type": "historical_ticket_description",
        }
    ]

    monkeypatch.setattr(
        tickets_routes, "search_similar_tickets", lambda *args, **kwargs: fake_results
    )

    response = client.post(
        "/api/v1/tickets/search",
        json={
            "ticket_id": "TEST-SEARCH-001",
            "query": "Outlook authentication fails after password change",
            "top_k": 5,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["result_count"] == 1
    assert body["results"][0]["ticket_id"] == "ITS-TRAIN-TEST-001"
    assert body["results"][0]["evidence_type"] == "historical_ticket_description"


def test_search_validates_top_k_limit(client):
    response = client.post(
        "/api/v1/tickets/search",
        json={"query": "Outlook cannot authenticate", "top_k": 50},
    )

    assert response.status_code == 422
