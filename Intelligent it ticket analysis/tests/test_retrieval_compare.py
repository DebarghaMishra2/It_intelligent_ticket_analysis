def test_compare_search_returns_all_three_methods(client, monkeypatch):
    from src.api.routes import tickets as tickets_routes

    fake_comparison = {
        "top_k": 3,
        "category_filter": None,
        "methods": {
            "bm25": {"result_count": 1, "results": [{"ticket_id": "BM25-1"}]},
            "semantic": {"result_count": 1, "results": [{"ticket_id": "SEM-1"}]},
            "hybrid": {"result_count": 1, "results": [{"ticket_id": "HYB-1"}]},
        },
        "score_comparison_note": "Do not compare raw scores across methods.",
        "filter_note": None,
        "evidence_note": "Historical descriptions are not verified resolutions.",
    }

    monkeypatch.setattr(
        tickets_routes,
        "compare_retrieval_methods",
        lambda *args, **kwargs: fake_comparison,
    )

    response = client.post(
        "/api/v1/tickets/search/compare",
        json={
            "ticket_id": "COMPARE-001",
            "query": "Outlook authentication fails after password reset",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["methods"]["bm25"]["results"][0]["ticket_id"] == "BM25-1"
    assert body["methods"]["semantic"]["results"][0]["ticket_id"] == "SEM-1"
    assert body["methods"]["hybrid"]["results"][0]["ticket_id"] == "HYB-1"
    assert body["request_id"]


def test_compare_service_builds_bm25_semantic_and_hybrid_lists(monkeypatch):
    from src.api import services

    bm25_fixture = [
        {
            "ticket_id": "HIST-1",
            "category": "O365",
            "description": "Outlook login fails",
            "search_text": "outlook login fails",
            "bm25_score": 8.5,
        }
    ]
    semantic_fixture = [
        {
            "ticket_id": "HIST-2",
            "category": "O365",
            "description": "Password reset affects Outlook",
            "search_text": "password reset affects outlook",
            "semantic_score": 0.82,
        }
    ]

    monkeypatch.setattr(services, "search_bm25", lambda *args, **kwargs: bm25_fixture)
    monkeypatch.setattr(
        services, "search_semantic", lambda *args, **kwargs: semantic_fixture
    )
    monkeypatch.setattr(
        services,
        "reciprocal_rank_fusion",
        lambda bm25, semantic, **kwargs: [
            {
                **bm25[0],
                "bm25_rank": 1,
                "semantic_rank": 1,
                "semantic_score": semantic[0]["semantic_score"],
                "rrf_score": 0.032,
            },
            {
                **semantic[0],
                "bm25_rank": None,
                "semantic_rank": 1,
                "bm25_score": None,
                "rrf_score": 0.016,
            },
        ],
    )

    resources = {
        "bm25": object(),
        "bm25_df": object(),
        "embedding_model": object(),
        "embeddings": object(),
        "semantic_metadata": object(),
    }

    result = services.compare_retrieval_methods(
        "Outlook login fails after password reset", resources, top_k=5
    )

    assert result["methods"]["bm25"]["results"][0]["ticket_id"] == "HIST-1"
    assert result["methods"]["semantic"]["results"][0]["ticket_id"] == "HIST-2"
    assert len(result["methods"]["hybrid"]["results"]) == 2
    assert result["methods"]["bm25"]["results"][0]["retrieval_score_type"] == "bm25_score"
    assert result["methods"]["semantic"]["results"][0]["retrieval_score_type"] == "cosine_similarity"
