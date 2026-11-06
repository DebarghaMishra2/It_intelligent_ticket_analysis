"""Lightweight retrieval components for Render's constrained Free plan.

This module deliberately avoids Sentence Transformers, PyTorch, and Hugging Face.
It uses BM25 plus scikit-learn TF-IDF cosine similarity over the same indexed
historical ticket descriptions. TF-IDF is a lightweight lexical baseline; it is
not a multilingual neural embedding model and should not be described as such.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

try:
    from src.preprocessing.text_cleaner import normalize_ticket_text
except ImportError:  # Allows this small module's unit tests to run in isolation.
    import re
    import unicodedata

    def normalize_ticket_text(value: str) -> str:
        value = unicodedata.normalize("NFKC", str(value or "")).lower()
        return re.sub(r"\s+", " ", value).strip()


def build_tfidf_index(metadata) -> Tuple[TfidfVectorizer, Any]:
    """Fit a compact TF-IDF index against the BM25 dataframe's search_text."""
    if "search_text" not in metadata.columns:
        raise ValueError("The historical ticket dataframe must contain 'search_text'.")

    documents = metadata["search_text"].fillna("").astype(str).tolist()
    if not documents or not any(text.strip() for text in documents):
        raise ValueError("No searchable historical ticket text is available.")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        max_features=20000,
        min_df=1,
        sublinear_tf=True,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(documents)
    if matrix.shape[0] != len(metadata):
        raise ValueError("TF-IDF matrix rows do not match historical ticket metadata.")
    return vectorizer, matrix


def search_tfidf(
    query: str,
    vectorizer: TfidfVectorizer,
    matrix: Any,
    metadata,
    top_k: int = 50,
) -> List[Dict[str, Any]]:
    """Return top TF-IDF cosine matches with metadata and their real score name."""
    if not isinstance(query, str) or not query.strip():
        return []
    if top_k < 1:
        return []

    cleaned_query = normalize_ticket_text(query) or query.strip()
    query_vector = vectorizer.transform([cleaned_query])
    scores = linear_kernel(query_vector, matrix).ravel()
    if scores.size == 0:
        return []

    # Stable tie-breaking makes repeated requests reproducible.
    ranked_indices = np.argsort(-scores, kind="stable")[: min(top_k, len(scores))]
    results: List[Dict[str, Any]] = []
    for index in ranked_indices:
        score = float(scores[int(index)])
        if score <= 0.0:
            continue
        row = metadata.iloc[int(index)]
        results.append(
            {
                "ticket_id": str(row.get("ticket_id", "")),
                "category": str(row.get("category", "")),
                "description": str(row.get("description", "") or ""),
                "search_text": str(row.get("search_text", "") or ""),
                "tfidf_score": score,
            }
        )
    return results


def fuse_bm25_tfidf(
    bm25_results: List[Dict[str, Any]],
    tfidf_results: List[Dict[str, Any]],
    rrf_k: int = 60,
    top_k: int = 50,
) -> List[Dict[str, Any]]:
    """Fuse BM25 and TF-IDF rankings using Reciprocal Rank Fusion.

    Raw BM25 and TF-IDF scores are not added together because they have
    different scales. Only rank positions contribute to the RRF score.
    """
    combined: Dict[str, Dict[str, Any]] = {}

    for rank, result in enumerate(bm25_results, start=1):
        ticket_id = str(result["ticket_id"])
        combined[ticket_id] = {
            "ticket_id": ticket_id,
            "category": result.get("category"),
            "description": result.get("description", ""),
            "search_text": result.get("search_text", ""),
            "bm25_rank": rank,
            "bm25_score": float(result.get("bm25_score", 0.0)),
            "tfidf_rank": None,
            "tfidf_score": None,
            "rrf_score": 1.0 / (rrf_k + rank),
        }

    for rank, result in enumerate(tfidf_results, start=1):
        ticket_id = str(result["ticket_id"])
        if ticket_id not in combined:
            combined[ticket_id] = {
                "ticket_id": ticket_id,
                "category": result.get("category"),
                "description": result.get("description", ""),
                "search_text": result.get("search_text", ""),
                "bm25_rank": None,
                "bm25_score": None,
                "tfidf_rank": rank,
                "tfidf_score": float(result.get("tfidf_score", 0.0)),
                "rrf_score": 0.0,
            }
        else:
            combined[ticket_id]["tfidf_rank"] = rank
            combined[ticket_id]["tfidf_score"] = float(result.get("tfidf_score", 0.0))

        combined[ticket_id]["rrf_score"] += 1.0 / (rrf_k + rank)

    ranked = sorted(
        combined.values(),
        key=lambda item: (-item["rrf_score"], item["ticket_id"]),
    )
    return ranked[:top_k]
