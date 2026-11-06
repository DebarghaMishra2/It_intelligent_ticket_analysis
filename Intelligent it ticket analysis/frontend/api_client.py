"""IT-Sentinel client with HTTP and embedded Streamlit execution modes.

Set Streamlit Cloud secret IT_SENTINEL_MODE = "local" to run the lightweight
in-process profile. The default remains the existing FastAPI HTTP client.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional
from urllib.parse import quote

import requests


class APIError(RuntimeError):
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class ITSentinelClient:
    def __init__(self, base_url: str, timeout: int = 90):
        normalized_url = (base_url or "").strip().rstrip("/")
        mode = os.getenv("IT_SENTINEL_MODE", "").strip().lower()
        self.local_mode = normalized_url.lower() in {"local", "embedded", "streamlit"} or mode in {"local", "embedded", "streamlit"}
        self.timeout = timeout
        self.session = requests.Session()
        self.base_url = "local" if self.local_mode else normalized_url
        self._runtime = None
        if self.local_mode:
            try:
                from frontend.local_runtime import get_local_runtime
                self._runtime = get_local_runtime()
            except Exception as exc:
                raise APIError(f"Could not initialize embedded IT-Sentinel runtime: {exc}") from exc
        else:
            if not normalized_url:
                raise ValueError("API base URL cannot be empty unless local mode is enabled.")
            if "://" not in normalized_url:
                normalized_url = f"http://{normalized_url}"
            self.base_url = normalized_url

    def _request(self, method: str, path: str, **kwargs: Any) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        try:
            response = self.session.request(method, url, timeout=kwargs.pop("timeout", self.timeout), **kwargs)
        except requests.RequestException as exc:
            raise APIError(
                f"Could not connect to IT-Sentinel API at {self.base_url}. "
                "Check that Uvicorn is running and the API URL is correct. "
                f"Details: {exc}"
            ) from exc
        try:
            payload = response.json()
        except ValueError:
            payload = {"detail": response.text[:2000]}
        if not response.ok:
            detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
            raise APIError(str(detail), status_code=response.status_code)
        if not isinstance(payload, dict):
            raise APIError("The API returned an unexpected response format.", response.status_code)
        return payload

    def _local_call(self, method_name: str, payload: Optional[Dict[str, Any]] = None, value: Optional[str] = None) -> Dict[str, Any]:
        if self._runtime is None:
            raise APIError("Embedded IT-Sentinel runtime is not initialized.")
        try:
            method = getattr(self._runtime, method_name)
            return method(payload) if payload is not None else method(value) if value is not None else method()
        except (ValueError, LookupError, KeyError) as exc:
            raise APIError(str(exc), status_code=400 if isinstance(exc, ValueError) else 404) from exc
        except APIError:
            raise
        except Exception as exc:
            raise APIError(f"Embedded runtime operation '{method_name}' failed: {exc}") from exc

    def health(self) -> Dict[str, Any]:
        return self._local_call("health") if self.local_mode else self._request("GET", "/health", timeout=5)

    def analyze_ticket(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._local_call("analyze_ticket", payload) if self.local_mode else self._request("POST", "/api/v1/tickets/analyze", json=payload)

    def search_tickets(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._local_call("search_tickets", payload) if self.local_mode else self._request("POST", "/api/v1/tickets/search", json=payload)

    def compare_retrieval(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._local_call("compare_retrieval", payload) if self.local_mode else self._request("POST", "/api/v1/tickets/search/compare", json=payload)

    def recommend(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._local_call("recommend", payload) if self.local_mode else self._request("POST", "/api/v1/tickets/recommend", json=payload)

    def submit_feedback(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._local_call("submit_feedback", payload) if self.local_mode else self._request("POST", "/api/v1/feedback", json=payload)

    def audit_ticket(self, ticket_id: str) -> Dict[str, Any]:
        return self._local_call("audit_ticket", value=ticket_id) if self.local_mode else self._request("GET", f"/api/v1/audit/tickets/{quote(ticket_id, safe='')}")

    def audit_request(self, request_id: str) -> Dict[str, Any]:
        return self._local_call("audit_request", value=request_id) if self.local_mode else self._request("GET", f"/api/v1/audit/requests/{quote(request_id, safe='')}")
