"""Lightweight in-process runtime for IT-Sentinel on Streamlit Community Cloud.

This profile intentionally avoids PyTorch/Sentence Transformers. It uses the
existing trained scikit-learn classifier, existing BM25 index builder, and a
TF-IDF cosine-similarity fallback for the second retrieval signal. The UI must
label that signal as TF-IDF, not as neural semantic search.

The full FastAPI + Sentence Transformer implementation remains unchanged for
local/Docker deployments.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.classification.predict import load_model, predict_ticket
from src.preprocessing.priority_rules import assign_priority
from src.retrieval.bm25_search import build_bm25_index, search_bm25

ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_PATH = ROOT / "data" / "knowledge_base" / "approved_articles.json"
DEFAULT_DB_PATH = Path(os.getenv("IT_SENTINEL_DATABASE_PATH", "/tmp/it_sentinel_streamlit.db"))
REVIEW_THRESHOLD = float(os.getenv("IT_SENTINEL_CLASSIFICATION_REVIEW_THRESHOLD", "0.60"))
RRF_K = 60


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def _tokens(text: str) -> set[str]:
    import re
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 1}


class AuditStore:
    """Small SQLite audit store. On Streamlit Cloud its filesystem is ephemeral."""

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=20)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS audit_requests (
                    request_id TEXT PRIMARY KEY,
                    ticket_id TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    input_text TEXT,
                    output_json TEXT NOT NULL
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS analyst_feedback (
                    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    ticket_id TEXT,
                    feedback TEXT NOT NULL,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(request_id) REFERENCES audit_requests(request_id)
                )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_ticket ON audit_requests(ticket_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_request ON analyst_feedback(request_id)")

    def save_request(self, endpoint: str, ticket_id: str, input_text: str, output: Dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO audit_requests VALUES (?, ?, ?, ?, ?, ?)",
                (
                    output["request_id"], ticket_id, endpoint, _now(), input_text,
                    json.dumps(output, ensure_ascii=False, default=str),
                ),
            )

    def save_feedback(self, request_id: str, ticket_id: Optional[str], feedback: str, notes: Optional[str]) -> Dict[str, Any]:
        with self._connect() as conn:
            exists = conn.execute(
                "SELECT ticket_id FROM audit_requests WHERE request_id = ?", (request_id,)
            ).fetchone()
            if exists is None:
                raise KeyError("No audit request was found for this request_id.")
            actual_ticket_id = ticket_id or exists["ticket_id"]
            created_at = _now()
            cursor = conn.execute(
                "INSERT INTO analyst_feedback (request_id, ticket_id, feedback, notes, created_at) VALUES (?, ?, ?, ?, ?)",
                (request_id, actual_ticket_id, feedback, notes, created_at),
            )
            return {
                "status": "saved", "feedback_id": cursor.lastrowid,
                "request_id": request_id, "ticket_id": actual_ticket_id,
                "feedback": feedback, "notes": notes, "created_at": created_at,
            }

    def _feedback_for(self, request_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT feedback_id, feedback, notes, created_at FROM analyst_feedback WHERE request_id = ? ORDER BY feedback_id",
                (request_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_request(self, request_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM audit_requests WHERE request_id = ?", (request_id,)).fetchone()
        if row is None:
            raise KeyError("Audit request not found.")
        item = dict(row)
        feedback = self._feedback_for(request_id)
        item["output"] = json.loads(item.pop("output_json"))
        item["feedback_history"] = feedback
        item["analyst_decision"] = feedback[-1]["feedback"] if feedback else None
        item["analyst_notes"] = feedback[-1]["notes"] if feedback else None
        return item

    def get_ticket(self, ticket_id: str) -> Dict[str, Any]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT request_id FROM audit_requests WHERE ticket_id = ? ORDER BY created_at DESC",
                (ticket_id,),
            ).fetchall()
        records = []
        for row in rows:
            try:
                record = self.get_request(row["request_id"])
            except KeyError:
                continue
            record.pop("feedback_history", None)
            records.append(record)
        return {"ticket_id": ticket_id, "record_count": len(records), "records": records}


class LocalITSentinelRuntime:
    """Loads the existing classifier and BM25 corpus once per Streamlit process."""

    def __init__(self) -> None:
        self.model = load_model()
        self.bm25, self.train_df = build_bm25_index()
        if self.train_df is None or self.train_df.empty:
            raise RuntimeError("The BM25 training corpus is empty.")

        self.train_df = self.train_df.reset_index(drop=True).copy()
        if "search_text" not in self.train_df.columns:
            self.train_df["search_text"] = self.train_df.get("description", "").fillna("").astype(str)

        # Lightweight lexical similarity. It is deliberately not called neural semantic search.
        self.tfidf = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            max_features=60000,
            sublinear_tf=True,
            norm="l2",
        )
        corpus = self.train_df["search_text"].fillna("").astype(str).tolist()
        self.tfidf_matrix = self.tfidf.fit_transform(corpus)
        self.audit = AuditStore()
        self.knowledge_articles = self._load_knowledge_articles()

    @property
    def models_loaded(self) -> bool:
        return self.model is not None and self.bm25 is not None and self.tfidf_matrix.shape[0] > 0

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok" if self.models_loaded else "starting",
            "service": "IT-Sentinel Streamlit Runtime",
            "models_loaded": bool(self.models_loaded),
            "execution_mode": "streamlit_embedded",
            "historical_ticket_count": int(len(self.train_df)),
            "retrieval_profile": "BM25 + TF-IDF + RRF (lightweight profile)",
        }

    def _ticket_id(self, payload: Dict[str, Any]) -> str:
        value = _safe_text(payload.get("ticket_id"))
        return value or f"STREAMLIT-{uuid.uuid4().hex[:10].upper()}"

    def _save(self, endpoint: str, ticket_id: str, input_text: str, result: Dict[str, Any]) -> Dict[str, Any]:
        self.audit.save_request(endpoint, ticket_id, input_text, result)
        return result

    def analyze_ticket(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        description = _safe_text(payload.get("description"))
        if len(description) < 5:
            raise ValueError("Ticket description must contain at least five characters.")
        prediction = predict_ticket(description, self.model)
        probability = float(prediction.get("model_probability", 0.0))
        confidence_band = "low" if probability < REVIEW_THRESHOLD else ("medium" if probability < 0.80 else "high")
        impact = _safe_text(payload.get("impact")).lower()
        urgency = _safe_text(payload.get("urgency")).lower()
        priority = assign_priority(impact, urgency) if impact in {"low", "medium", "high"} and urgency in {"low", "medium", "high"} else None
        ticket_id = self._ticket_id(payload)
        result = {
            "request_id": str(uuid.uuid4()),
            "ticket_id": ticket_id,
            "classification": prediction,
            "confidence_band": confidence_band,
            "human_review_recommended": probability < REVIEW_THRESHOLD,
            "review_threshold": REVIEW_THRESHOLD,
            "priority": priority,
            "impact": impact or None,
            "urgency": urgency or None,
            "priority_note": "Priority is based on the project's demonstration impact/urgency matrix; it is not an official HCLTech SLA or production priority policy." if priority else None,
            "limitations": [
                "Model probability is a classifier confidence estimate, not a guarantee of correctness.",
                "The priority matrix is a project demonstration rule and must be replaced by an approved policy before production use.",
            ],
        }
        return self._save("/api/v1/tickets/analyze", ticket_id, description, result)

    def _rank_bm25(self, query: str) -> List[Dict[str, Any]]:
        raw = search_bm25(query, self.bm25, self.train_df, top_k=len(self.train_df))
        output = []
        for rank, item in enumerate(raw, start=1):
            row = dict(item)
            row["ticket_id"] = str(row.get("ticket_id", ""))
            row["bm25_rank"] = rank
            row["bm25_score"] = float(row.get("bm25_score", 0.0))
            output.append(row)
        return output

    def _rank_tfidf(self, query: str) -> List[Dict[str, Any]]:
        query_vector = self.tfidf.transform([query])
        scores = (self.tfidf_matrix @ query_vector.T).toarray().ravel()
        order = np.argsort(-scores, kind="stable")
        output = []
        for rank, idx in enumerate(order, start=1):
            source = self.train_df.iloc[int(idx)]
            score = float(scores[int(idx)])
            output.append({
                "ticket_id": str(source.get("ticket_id", "")),
                "category": str(source.get("category", "Unclassified")),
                "description": _safe_text(source.get("description")),
                "search_text": _safe_text(source.get("search_text")),
                "semantic_rank": rank,
                "semantic_score": score,
                "tfidf_score": score,
            })
        return output

    @staticmethod
    def _filter_category(items: List[Dict[str, Any]], category: Optional[str]) -> List[Dict[str, Any]]:
        if not category:
            return items
        return [item for item in items if str(item.get("category", "")).casefold() == category.casefold()]

    @staticmethod
    def _rrf(bm25_results: List[Dict[str, Any]], tfidf_results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        combined: Dict[str, Dict[str, Any]] = {}
        for rank, item in enumerate(bm25_results, start=1):
            tid = str(item["ticket_id"])
            combined[tid] = {
                "ticket_id": tid, "category": item.get("category"),
                "description": item.get("description"), "search_text": item.get("search_text"),
                "bm25_rank": rank, "bm25_score": item.get("bm25_score"),
                "semantic_rank": None, "semantic_score": None,
                "rrf_score": 1.0 / (RRF_K + rank),
            }
        for rank, item in enumerate(tfidf_results, start=1):
            tid = str(item["ticket_id"])
            if tid not in combined:
                combined[tid] = {
                    "ticket_id": tid, "category": item.get("category"),
                    "description": item.get("description"), "search_text": item.get("search_text"),
                    "bm25_rank": None, "bm25_score": None,
                    "semantic_rank": rank, "semantic_score": item.get("semantic_score"),
                    "rrf_score": 0.0,
                }
            else:
                combined[tid]["semantic_rank"] = rank
                combined[tid]["semantic_score"] = item.get("semantic_score")
            combined[tid]["rrf_score"] += 1.0 / (RRF_K + rank)
        ordered = sorted(combined.values(), key=lambda x: (-x["rrf_score"], x["ticket_id"]))
        return ordered[:top_k]

    def search_tickets(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = _safe_text(payload.get("query"))
        if len(query) < 3:
            raise ValueError("Search query must contain at least three characters.")
        top_k = max(1, min(int(payload.get("top_k", 5)), 10))
        category = _safe_text(payload.get("category") or payload.get("category_filter")) or None
        bm25_all = self._rank_bm25(query)
        tfidf_all = self._rank_tfidf(query)
        hybrid_all = self._rrf(bm25_all, tfidf_all, top_k=len(self.train_df))
        hybrid_results = self._filter_category(hybrid_all, category)[:top_k]
        results = []
        for rank, item in enumerate(hybrid_results[:top_k], start=1):
            results.append({**item, "rank": rank, "evidence_type": "historical_ticket_description"})
        ticket_id = self._ticket_id(payload)
        result = {
            "request_id": str(uuid.uuid4()), "ticket_id": ticket_id, "query": query,
            "category_filter": category, "result_count": len(results), "results": results,
            "filter_note": "Category filter applied after retrieving candidates." if category else None,
            "retrieval_profile": "Hybrid RRF over BM25 + TF-IDF; this free-hosted profile does not run the Sentence Transformer model.",
        }
        return self._save("/api/v1/tickets/search", ticket_id, query, result)

    def compare_retrieval(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query = _safe_text(payload.get("query"))
        if len(query) < 3:
            raise ValueError("Search query must contain at least three characters.")
        top_k = max(1, min(int(payload.get("top_k", 5)), 10))
        category = _safe_text(payload.get("category") or payload.get("category_filter")) or None
        bm25_all = self._filter_category(self._rank_bm25(query), category)
        tfidf_all = self._filter_category(self._rank_tfidf(query), category)
        hybrid_all = self._filter_category(self._rrf(bm25_all, tfidf_all, len(self.train_df)), category)

        bm25_rows, tfidf_rows, hybrid_rows = bm25_all[:top_k], tfidf_all[:top_k], hybrid_all[:top_k]
        methods = {
            "bm25": {"result_count": len(bm25_rows), "results": []},
            "semantic": {"result_count": len(tfidf_rows), "results": []},
            "hybrid": {"result_count": len(hybrid_rows), "results": []},
        }
        for key, rows, score_key, score_type in [
            ("bm25", bm25_rows, "bm25_score", "bm25_score"),
            ("semantic", tfidf_rows, "semantic_score", "tfidf_cosine_similarity"),
            ("hybrid", hybrid_rows, "rrf_score", "reciprocal_rank_fusion_score"),
        ]:
            formatted = []
            for rank, row in enumerate(rows, start=1):
                formatted.append({
                    **row, "rank": rank, "retrieval_method": "tfidf" if key == "semantic" else key,
                    "retrieval_score": row.get(score_key), "retrieval_score_type": score_type,
                    "evidence_type": "historical_ticket_description",
                })
            methods[key]["results"] = formatted
        ticket_id = self._ticket_id(payload)
        result = {
            "request_id": str(uuid.uuid4()), "ticket_id": ticket_id, "query": query,
            "top_k": top_k, "category_filter": category, "methods": methods,
            "score_comparison_note": "BM25 and TF-IDF cosine similarity use different scales. Compare ranking positions and reviewed relevance, not raw scores across methods.",
            "evidence_note": "Historical ticket descriptions are not verified resolution steps.",
            "retrieval_profile": "Lightweight free-hosted profile: BM25 + TF-IDF + RRF; no Sentence Transformer inference.",
        }
        return self._save("/api/v1/tickets/search/compare", ticket_id, query, result)

    def _load_knowledge_articles(self) -> List[Dict[str, Any]]:
        if not KNOWLEDGE_PATH.exists():
            return []
        try:
            payload = json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if isinstance(payload, list):
            articles = payload
        elif isinstance(payload, dict):
            articles = payload.get("articles", [])
        else:
            articles = []
        return [item for item in articles if isinstance(item, dict)]

    def _find_approved_knowledge(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        matches = []
        for article in self.knowledge_articles:
            status = str(article.get("approval_status", article.get("status", ""))).casefold()
            if status != "approved" and article.get("is_approved") is not True:
                continue
            searchable = " ".join(_safe_text(article.get(key)) for key in (
                "title", "application", "category", "subcategory", "issue_type", "summary", "content", "description"
            ))
            article_tokens = _tokens(searchable)
            if not article_tokens:
                continue
            score = len(query_tokens & article_tokens) / max(1, len(query_tokens))
            steps = article.get("resolution_steps") or article.get("steps") or []
            if isinstance(steps, str):
                steps = [steps]
            if score < 0.15 or not steps:
                continue
            matches.append({
                "article_id": article.get("article_id", article.get("id")),
                "title": article.get("title", "Approved knowledge article"),
                "excerpt": _safe_text(article.get("summary") or article.get("content") or article.get("description"))[:600],
                "source_reference": article.get("source_url") or article.get("source_reference"),
                "approval_status": "approved", "match_score": round(score, 4),
                "steps": steps,
                "source_name": article.get("source_name") or article.get("title") or "Approved knowledge base",
            })
        return sorted(matches, key=lambda x: -x["match_score"])[:top_k]

    def recommend(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        description = _safe_text(payload.get("description") or payload.get("query"))
        if len(description) < 5:
            raise ValueError("Ticket description must contain at least five characters.")
        top_k = max(1, min(int(payload.get("top_k", 5)), 10))
        ticket_id = self._ticket_id(payload)
        # Generate precedents with the same ranking profile as the search endpoint.
        search_result = self.search_tickets({"ticket_id": ticket_id, "query": description, "top_k": top_k})
        matches = self._find_approved_knowledge(description, top_k=3)
        recommendation = None
        if matches:
            best = matches[0]
            recommendation = {
                "steps": best["steps"], "source_name": best["source_name"],
                "source_article_id": best.get("article_id"),
                "source_reference": best.get("source_reference"),
            }
        status = "knowledge_match_found" if recommendation else "insufficient_resolution_evidence"
        message = (
            "Approved knowledge candidates were found. An analyst must validate the article and steps before taking action."
            if recommendation else
            "Similar historical ticket descriptions were found, but no matching approved knowledge article with resolution steps was found. No fix is invented."
        )
        result = {
            "request_id": str(uuid.uuid4()), "ticket_id": ticket_id, "query": description,
            "status": status, "recommendation": recommendation,
            "approved_knowledge_matches": [
                {k: v for k, v in item.items() if k != "steps"} for item in matches
            ],
            "similar_historical_tickets": search_result.get("results", []),
            "human_review_required": True,
            "message": message,
            "safety_note": "IT-Sentinel provides decision support only; it does not close tickets or execute remediation.",
        }
        return self._save("/api/v1/tickets/recommend", ticket_id, description, result)

    def submit_feedback(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request_id = _safe_text(payload.get("request_id"))
        feedback = _safe_text(payload.get("feedback")).lower()
        if feedback not in {"useful", "incorrect", "unresolved"}:
            raise ValueError("Feedback must be useful, incorrect, or unresolved.")
        try:
            return self.audit.save_feedback(
                request_id, _safe_text(payload.get("ticket_id")) or None,
                feedback, _safe_text(payload.get("notes")) or None,
            )
        except KeyError as exc:
            raise LookupError(str(exc)) from exc

    def audit_ticket(self, ticket_id: str) -> Dict[str, Any]:
        return self.audit.get_ticket(ticket_id)

    def audit_request(self, request_id: str) -> Dict[str, Any]:
        return self.audit.get_request(request_id)


@lru_cache(maxsize=1)
def get_local_runtime() -> LocalITSentinelRuntime:
    """One model/index load per Streamlit process."""
    return LocalITSentinelRuntime()
