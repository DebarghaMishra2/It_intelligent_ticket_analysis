"""IT-Sentinel services with full and lightweight retrieval profiles.

Default/local profile:
    Sentence Transformer + BM25 + RRF.

Render Free profile (IT_SENTINEL_RETRIEVAL_MODE=bm25_tfidf):
    scikit-learn TF-IDF + BM25 + RRF; no PyTorch or Sentence Transformers.

The free profile intentionally changes retrieval semantics. TF-IDF is a
lightweight lexical similarity baseline, not a neural semantic embedding model.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from src.classification.predict import load_model, predict_ticket
from src.preprocessing.priority_rules import assign_priority
from src.retrieval.bm25_search import build_bm25_index, search_bm25
from src.api.free_retrieval import build_tfidf_index, search_tfidf, fuse_bm25_tfidf
from src.api.knowledge_base import load_approved_knowledge_base, search_approved_knowledge

CANDIDATE_K = 50


def search_semantic(
    query,
    embedding_model,
    embeddings,
    semantic_metadata,
    top_k=CANDIDATE_K,
):
    """
    Lazy compatibility wrapper for full Sentence Transformer retrieval.

    The heavyweight semantic module is imported only when the full
    retrieval profile actually calls this function.
    """
    from src.retrieval.semantic_search import (
        search_semantic as _search_semantic,
    )

    return _search_semantic(
        query,
        embedding_model,
        embeddings,
        semantic_metadata,
        top_k=top_k,
    )


LIGHTWEIGHT_MODES = {"bm25_tfidf", "tfidf", "lightweight", "free"}


def initialize_resources() -> Dict[str, Any]:
    """Load only the components required by the selected deployment profile."""
    classifier = load_model()
    bm25, bm25_df = build_bm25_index()
    knowledge_base = load_approved_knowledge_base()
    retrieval_mode = os.getenv("IT_SENTINEL_RETRIEVAL_MODE", "full").strip().lower()

    if retrieval_mode in LIGHTWEIGHT_MODES:
        vectorizer, tfidf_matrix = build_tfidf_index(bm25_df)
        return {
            "classifier": classifier,
            "bm25": bm25,
            "bm25_df": bm25_df,
            "tfidf_vectorizer": vectorizer,
            "tfidf_matrix": tfidf_matrix,
            "knowledge_base": knowledge_base,
            "retrieval_mode": "bm25_tfidf",
            "ready": True,
        }

    # Import the heavy semantic stack only in the full/local profile.
    from src.retrieval.hybrid_search import validate_shared_corpus
    from src.retrieval.semantic_search import (
        load_embedding_model,
        load_semantic_index,
    )

    embeddings, semantic_metadata = load_semantic_index()
    validate_shared_corpus(bm25_df, semantic_metadata)
    embedding_model = load_embedding_model()

    return {
        "classifier": classifier,
        "bm25": bm25,
        "bm25_df": bm25_df,
        "embeddings": embeddings,
        "semantic_metadata": semantic_metadata,
        "embedding_model": embedding_model,
        "knowledge_base": knowledge_base,
        "retrieval_mode": "sentence_transformer",
        "ready": True,
    }


def _json_safe(value: Any) -> Any:
    """Convert NumPy/Pandas values into JSON-compatible Python values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (ValueError, TypeError):
            pass
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        import pandas as pd
        if pd.isna(value):
            return None
    except (ImportError, TypeError, ValueError):
        pass
    return str(value)


def classify_and_prioritize(
    description: str,
    resources: Dict[str, Any],
    impact: Optional[str] = None,
    urgency: Optional[str] = None,
) -> Dict[str, Any]:
    classification = predict_ticket(description, resources["classifier"])
    probability = float(classification["model_probability"])
    review_threshold = float(os.getenv("IT_SENTINEL_CLASSIFICATION_REVIEW_THRESHOLD", "0.60"))

    if probability >= 0.80:
        confidence_band = "high"
    elif probability >= review_threshold:
        confidence_band = "medium"
    else:
        confidence_band = "low"

    priority = None
    priority_note = "Impact and urgency were not both provided; priority was not assigned."
    if impact is not None and urgency is not None:
        priority = assign_priority(impact, urgency)
        priority_note = (
            "Priority is based on the project's demonstration impact/urgency matrix; "
            "it is not an official HCLTech SLA or production priority policy."
        )
    elif impact is not None or urgency is not None:
        priority_note = "Provide both impact and urgency to calculate a demo priority."

    return {
        "classification": _json_safe(classification),
        "confidence_band": confidence_band,
        "human_review_recommended": probability < review_threshold,
        "review_threshold": review_threshold,
        "priority": priority,
        "impact": impact,
        "urgency": urgency,
        "priority_note": priority_note,
        "limitations": [
            "Model probability is a classifier confidence estimate, not a guarantee of correctness.",
            "The priority matrix is a project demonstration rule and must be replaced by an approved policy before production use.",
        ],
    }


def _filter_results_by_category(
    results: List[Dict[str, Any]], category: Optional[str]
) -> List[Dict[str, Any]]:
    if not category:
        return results
    expected = category.casefold().strip()
    return [
        result for result in results
        if str(result.get("category", "")).casefold().strip() == expected
    ]


def _search_free_methods(query: str, resources: Dict[str, Any]):
    bm25_results = search_bm25(
        query, resources["bm25"], resources["bm25_df"], top_k=CANDIDATE_K
    )
    tfidf_results = search_tfidf(
        query,
        resources["tfidf_vectorizer"],
        resources["tfidf_matrix"],
        resources["bm25_df"],
        top_k=CANDIDATE_K,
    )
    hybrid_results = fuse_bm25_tfidf(
        bm25_results, tfidf_results, rrf_k=60, top_k=CANDIDATE_K
    )
    return bm25_results, tfidf_results, hybrid_results


def search_similar_tickets(
    query: str,
    resources: Dict[str, Any],
    top_k: int = 5,
    category: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if resources.get("retrieval_mode") == "bm25_tfidf":
        _, _, fused = _search_free_methods(query, resources)
        fused = _filter_results_by_category(fused, category)
        output = []
        for rank, result in enumerate(fused[:top_k], start=1):
            output.append({
                "rank": rank,
                "ticket_id": str(result.get("ticket_id", "")),
                "category": _json_safe(result.get("category")),
                "description": str(result.get("description") or "")[:1500],
                "bm25_rank": result.get("bm25_rank"),
                "bm25_score": _json_safe(result.get("bm25_score")),
                "tfidf_rank": result.get("tfidf_rank"),
                "tfidf_score": _json_safe(result.get("tfidf_score")),
                "rrf_score": round(float(result.get("rrf_score", 0.0)), 8),
                "retrieval_mode": "BM25 + TF-IDF (lightweight)",
                "evidence_type": "historical_ticket_description",
            })
        return output

    # Preserve the existing full Sentence Transformer path for local/paid use.
    from src.retrieval.hybrid_search import reciprocal_rank_fusion

    bm25_results = search_bm25(
        query, resources["bm25"], resources["bm25_df"], top_k=CANDIDATE_K
    )
    semantic_results = search_semantic(
        query,
        resources["embedding_model"],
        resources["embeddings"],
        resources["semantic_metadata"],
        top_k=CANDIDATE_K,
    )
    fused = reciprocal_rank_fusion(
        bm25_results, semantic_results, top_k=CANDIDATE_K
    )
    fused = _filter_results_by_category(fused, category)

    output: List[Dict[str, Any]] = []
    for rank, result in enumerate(fused[:top_k], start=1):
        output.append({
            "rank": rank,
            "ticket_id": str(result.get("ticket_id", "")),
            "category": _json_safe(result.get("category")),
            "description": str(result.get("description") or "")[:1500],
            "bm25_rank": result.get("bm25_rank"),
            "bm25_score": _json_safe(result.get("bm25_score")),
            "semantic_rank": result.get("semantic_rank"),
            "semantic_score": _json_safe(result.get("semantic_score")),
            "rrf_score": round(float(result.get("rrf_score", 0.0)), 8),
            "retrieval_mode": "BM25 + Sentence Transformer",
            "evidence_type": "historical_ticket_description",
        })
    return output


def build_recommendation(
    description: str, resources: Dict[str, Any], top_k: int = 5
) -> Dict[str, Any]:
    similar_tickets = search_similar_tickets(description, resources, top_k=top_k)
    knowledge_matches = search_approved_knowledge(
        description, resources["knowledge_base"], top_k=3
    )

    if knowledge_matches:
        status = "approved_knowledge_candidates_found"
        recommendation = {
            "steps": knowledge_matches[0]["resolution_steps"],
            "source_article_id": knowledge_matches[0]["article_id"],
            "source_reference": knowledge_matches[0]["source_reference"],
            "source_name": knowledge_matches[0]["source_name"],
        }
        message = (
            "Potential steps were retrieved from an approved knowledge article. "
            "An analyst must validate applicability before acting."
        )
    else:
        status = "insufficient_resolution_evidence"
        recommendation = None
        message = (
            "Similar historical ticket descriptions were found, but the current "
            "historical corpus does not expose verified resolution text and no "
            "matching approved knowledge article was found. No fix is invented."
        )

    return {
        "status": status,
        "recommendation": recommendation,
        "approved_knowledge_matches": knowledge_matches,
        "similar_historical_tickets": similar_tickets,
        "human_review_required": True,
        "message": message,
        "safety_note": "IT-Sentinel provides decision support only; it does not close tickets or execute remediation.",
    }


def _format_method_results(
    results: List[Dict[str, Any]], method: str, top_k: int
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for rank, result in enumerate(results[:top_k], start=1):
        item = {
            "rank": rank,
            "ticket_id": str(result.get("ticket_id", "")),
            "category": _json_safe(result.get("category")),
            "description": str(result.get("description") or "")[:1500],
            "evidence_type": "historical_ticket_description",
        }
        if method == "bm25":
            item.update({
                "retrieval_method": "bm25",
                "retrieval_score": _json_safe(result.get("bm25_score")),
                "retrieval_score_type": "bm25_score",
                "bm25_rank": rank,
                "bm25_score": _json_safe(result.get("bm25_score")),
            })
        elif method == "tfidf":
            item.update({
                "retrieval_method": "tfidf",
                "retrieval_score": _json_safe(result.get("tfidf_score")),
                "retrieval_score_type": "tfidf_cosine_similarity",
                "tfidf_rank": rank,
                "tfidf_score": _json_safe(result.get("tfidf_score")),
            })
        elif method == "semantic":
            item.update({
                "retrieval_method": "semantic",
                "retrieval_score": _json_safe(result.get("semantic_score")),
                "retrieval_score_type": "sentence_transformer_cosine_similarity",
                "semantic_rank": rank,
                "semantic_score": _json_safe(result.get("semantic_score")),
            })
        else:
            item.update({
                "retrieval_method": "hybrid",
                "retrieval_score": _json_safe(result.get("rrf_score")),
                "retrieval_score_type": "reciprocal_rank_fusion_score",
                "bm25_rank": result.get("bm25_rank"),
                "bm25_score": _json_safe(result.get("bm25_score")),
                "rrf_score": round(float(result.get("rrf_score", 0.0)), 8),
            })
            if "tfidf_rank" in result:
                item["tfidf_rank"] = result.get("tfidf_rank")
                item["tfidf_score"] = _json_safe(result.get("tfidf_score"))
            if "semantic_rank" in result:
                item["semantic_rank"] = result.get("semantic_rank")
                item["semantic_score"] = _json_safe(result.get("semantic_score"))
        return_item = _json_safe(item)
        output.append(return_item)
    return output


def compare_retrieval_methods(
    query: str,
    resources: Dict[str, Any],
    top_k: int = 5,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare retrieval strategies available in the selected profile."""
    if resources.get("retrieval_mode") == "bm25_tfidf":
        bm25_results, tfidf_results, hybrid_results = _search_free_methods(query, resources)
        bm25_results = _filter_results_by_category(bm25_results, category)
        tfidf_results = _filter_results_by_category(tfidf_results, category)
        hybrid_results = _filter_results_by_category(hybrid_results, category)
        return {
            "top_k": top_k,
            "category_filter": category,
            "retrieval_mode": "bm25_tfidf",
            "method_labels": {
                "bm25": "BM25",
                "tfidf": "TF-IDF (lightweight)",
                "hybrid": "Hybrid RRF",
            },
            "methods": {
                "bm25": {
                    "result_count": min(len(bm25_results), top_k),
                    "results": _format_method_results(bm25_results, "bm25", top_k),
                },
                "tfidf": {
                    "result_count": min(len(tfidf_results), top_k),
                    "results": _format_method_results(tfidf_results, "tfidf", top_k),
                },
                "hybrid": {
                    "result_count": min(len(hybrid_results), top_k),
                    "results": _format_method_results(hybrid_results, "hybrid", top_k),
                },
            },
            "score_comparison_note": (
                "BM25 and TF-IDF cosine scores use different scales. Compare rank "
                "positions and human-reviewed relevance, not raw scores across methods."
            ),
            "filter_note": (
                "Category filtering is applied after each method retrieves its top-50 "
                "candidate pool; this is not a full-corpus filtered search."
                if category else None
            ),
            "evidence_note": (
                "The free profile uses BM25 and lightweight TF-IDF. Historical results "
                "are ticket descriptions, not verified resolution steps."
            ),
        }

    # Full local/paid profile: load Sentence Transformer imports only here.
    from src.retrieval.hybrid_search import reciprocal_rank_fusion

    bm25_results = search_bm25(
        query, resources["bm25"], resources["bm25_df"], top_k=CANDIDATE_K
    )
    semantic_results = search_semantic(
        query,
        resources["embedding_model"],
        resources["embeddings"],
        resources["semantic_metadata"],
        top_k=CANDIDATE_K,
    )
    hybrid_results = reciprocal_rank_fusion(
        bm25_results, semantic_results, top_k=CANDIDATE_K
    )
    bm25_results = _filter_results_by_category(bm25_results, category)
    semantic_results = _filter_results_by_category(semantic_results, category)
    hybrid_results = _filter_results_by_category(hybrid_results, category)

    return {
        "top_k": top_k,
        "category_filter": category,
        "retrieval_mode": "sentence_transformer",
        "method_labels": {
            "bm25": "BM25",
            "semantic": "Sentence Transformer",
            "hybrid": "Hybrid RRF",
        },
        "methods": {
            "bm25": {
                "result_count": min(len(bm25_results), top_k),
                "results": _format_method_results(bm25_results, "bm25", top_k),
            },
            "semantic": {
                "result_count": min(len(semantic_results), top_k),
                "results": _format_method_results(semantic_results, "semantic", top_k),
            },
            "hybrid": {
                "result_count": min(len(hybrid_results), top_k),
                "results": _format_method_results(hybrid_results, "hybrid", top_k),
            },
        },
        "score_comparison_note": (
            "BM25, Sentence Transformer cosine-similarity and reciprocal-rank-fusion "
            "scores use different scales. Compare ranking positions and reviewed "
            "relevance, not raw scores across methods."
        ),
        "filter_note": (
            "Category filtering is applied after each method retrieves its top-50 "
            "candidate pool; this is not a full-corpus filtered search."
            if category else None
        ),
        "evidence_note": (
            "Historical results contain ticket descriptions, not verified resolution steps."
        ),
    }
