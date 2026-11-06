"""Shared fixtures for IT-Sentinel API tests.

The suite uses a lightweight fake classifier and a temporary SQLite database.
It does not load Sentence Transformers, real embeddings, or production data.
"""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import database, services
from src.api.main import app


class DummyClassifier:
    """Small deterministic stand-in for the trained classifier."""

    classes_ = np.array(
        [
            "O365",
            "Support general",
            "Active Directory",
            "Fileservice",
            "Software",
            "EOL",
            "Computer-Services",
        ]
    )

    def predict(self, descriptions):
        return np.array(["O365" for _ in descriptions])

    def predict_proba(self, descriptions):
        probabilities = np.array(
            [[0.55, 0.20, 0.10, 0.06, 0.04, 0.03, 0.02]]
        )
        return np.repeat(probabilities, repeats=len(descriptions), axis=0)


@pytest.fixture

def client(tmp_path, monkeypatch):
    """Create an isolated API client with a temporary audit database."""
    test_database = tmp_path / "it_sentinel_test.db"
    monkeypatch.setattr(database, "DATABASE_PATH", test_database)
    database.initialize_database()

    app.state.resources = {
        "ready": True,
        "classifier": DummyClassifier(),
    }

    # The production `predict_ticket` function expects a fitted sklearn model.
    # Stub the imported service function so API tests remain deterministic.
    monkeypatch.setattr(
        services,
        "predict_ticket",
        lambda description, model: {
            "ticket_description": description,
            "predicted_category": "O365",
            "model_probability": 0.55,
            "top_categories": [
                {"category": "O365", "probability": 0.55},
                {"category": "Support general", "probability": 0.20},
                {"category": "Active Directory", "probability": 0.10},
            ],
        },
    )

    # Do not use `with TestClient(app)`: entering the lifespan would load the
    # real model and retrieval indexes. These tests exercise routes with fakes.
    test_client = TestClient(app)
    yield test_client
    test_client.close()
    app.state.resources = None
