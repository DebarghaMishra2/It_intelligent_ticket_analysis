def test_health_reports_ready(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "IT-Sentinel API"
    assert body["models_loaded"] is True


def test_health_reports_starting_when_resources_are_missing(client):
    from src.api.main import app

    app.state.resources = None
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "starting"
    assert response.json()["models_loaded"] is False
