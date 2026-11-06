from unittest.mock import Mock

import requests

from frontend.api_client import APIError, ITSentinelClient


def test_health_uses_health_endpoint(monkeypatch):
    response = Mock()
    response.ok = True
    response.json.return_value = {"status": "ok", "models_loaded": True}
    request_mock = Mock(return_value=response)
    monkeypatch.setattr(requests.Session, "request", request_mock)

    client = ITSentinelClient("http://127.0.0.1:8001/")
    result = client.health()

    assert result["status"] == "ok"
    request_mock.assert_called_once_with(
        "GET", "http://127.0.0.1:8001/health", timeout=5
    )


def test_api_error_includes_http_detail(monkeypatch):
    response = Mock()
    response.ok = False
    response.status_code = 422
    response.json.return_value = {"detail": "Description is too short"}
    monkeypatch.setattr(requests.Session, "request", Mock(return_value=response))

    client = ITSentinelClient("http://127.0.0.1:8001")
    try:
        client.health()
    except APIError as error:
        assert error.status_code == 422
        assert "Description is too short" in str(error)
    else:
        raise AssertionError("Expected APIError")
