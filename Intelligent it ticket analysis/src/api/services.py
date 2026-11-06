"""Application services that connect the existing ML and retrieval modules."""

import os
from typing import Any, Dict, List, Optional

from src.classification.predict import load_model, predict_ticket
from src.preprocessing.priority_rules import assign_priority
from src.retrieval.bm25_search import build_bm25_index, search_bm25
from src.retrieval.hybrid_search import reciprocal_rank_fusion, validate_shared_corpus
from src.retrieval.semantic_search import (
    load_embedding_model,
    load_semantic_index,
    search_semantic,
)
from src.api.knowledge_base import load_approved_knowledge_base, search_approved_knowledge

CANDIDATE_K = 50


def initialize_resources() -> Dict[str, Any]:
    """Load all model/index resources once during API startup."""
    classifier = load_model()
    bm25, bm25_df = build_bm25_index()
    embeddings, semantic_metadata = load_semantic_index()
    validate_shared_corpus(bm25_df, semantic_metadata)
    embedding_model = load_embedding_model()
    knowledge_base = load_approved_knowledge_base()

    return {
        "classifier": classifier,
        "bm25": bm25,
        "bm25_df": bm25_df,
        "embeddings": embeddings,
        "semantic_metadata": semantic_metadata,
        "embedding_model": embedding_model,
        "knowledge_base": knowledge_base,
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


def search_similar_tickets(
    query: str,
    resources: Dict[str, Any],
    top_k: int = 5,
    category: Optional[str] = None,
) -> List[Dict[str, Any]]:
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
        bm25_results,
        semantic_results,
        top_k=CANDIDATE_K,
    )

    if category:
        expected = category.casefold().strip()
        fused = [
            result for result in fused
            if str(result.get("category", "")).casefold().strip() == expected
        ]

    output: List[Dict[str, Any]] = []
    for rank, result in enumerate(fused[:top_k], start=1):
        description = result.get("description")
        if description is None:
            description = ""
        output.append(
            {
                "rank": rank,
                "ticket_id": str(result.get("ticket_id", "")),
                "category": _json_safe(result.get("category")),
                "description": str(description)[:1500],
                "bm25_rank": result.get("bm25_rank"),
                "bm25_score": _json_safe(result.get("bm25_score")),
                "semantic_rank": result.get("semantic_rank"),
                "semantic_score": _json_safe(result.get("semantic_score")),
                "rrf_score": round(float(result.get("rrf_score", 0.0)), 8),
                "evidence_type": "historical_ticket_description",
            }
        )
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


def _filter_results_by_category(
    results: List[Dict[str, Any]], category: Optional[str]
) -> List[Dict[str, Any]]:
    """Apply the existing exact category filter to a retrieved candidate pool."""
    if not category:
        return results
    expected = category.casefold().strip()
    return [
        result
        for result in results
        if str(result.get("category", "")).casefold().strip() == expected
    ]


def _format_method_results(
    results: List[Dict[str, Any]], method: str, top_k: int
) -> List[Dict[str, Any]]:
    """Normalize method-specific results into one comparable response shape."""
    output: List[Dict[str, Any]] = []

    for rank, result in enumerate(results[:top_k], start=1):
        bm25_rank = result.get("bm25_rank")
        semantic_rank = result.get("semantic_rank")
        bm25_score = result.get("bm25_score")
        semantic_score = result.get("semantic_score")
        rrf_score = result.get("rrf_score")

        if method == "bm25":
            bm25_rank = rank
            score = bm25_score
            score_type = "bm25_score"
        elif method == "semantic":
            semantic_rank = rank
            score = semantic_score
            score_type = "cosine_similarity"
        else:
            score = rrf_score
            score_type = "reciprocal_rank_fusion_score"

        output.append(
            {
                "rank": rank,
                "ticket_id": str(result.get("ticket_id", "")),
                "category": _json_safe(result.get("category")),
                "description": str(result.get("description") or "")[:1500],
                "retrieval_method": method,
                "retrieval_score": _json_safe(score),
                "retrieval_score_type": score_type,
                "bm25_rank": bm25_rank,
                "bm25_score": _json_safe(bm25_score),
                "semantic_rank": semantic_rank,
                "semantic_score": _json_safe(semantic_score),
                "rrf_score": (
                    round(float(rrf_score), 8) if rrf_score is not None else None
                ),
                "evidence_type": "historical_ticket_description",
            }
        )

    return output


def compare_retrieval_methods(
    query: str,
    resources: Dict[str, Any],
    top_k: int = 5,
    category: Optional[str] = None,
) -> Dict[str, Any]:
    """Return BM25, semantic, and hybrid rankings for retrieval diagnostics.

    Raw scores belong to different scales and must not be compared directly.
    Category filtering is applied to each method's top-CANDIDATE_K pool, not
    before searching the entire corpus.
    """
    bm25_results = search_bm25(
        query,
        resources["bm25"],
        resources["bm25_df"],
        top_k=CANDIDATE_K,
    )
    semantic_results = search_semantic(
        query,
        resources["embedding_model"],
        resources["embeddings"],
        resources["semantic_metadata"],
        top_k=CANDIDATE_K,
    )
    hybrid_results = reciprocal_rank_fusion(
        bm25_results,
        semantic_results,
        top_k=CANDIDATE_K,
    )

    bm25_results = _filter_results_by_category(bm25_results, category)
    semantic_results = _filter_results_by_category(semantic_results, category)
    hybrid_results = _filter_results_by_category(hybrid_results, category)

    return {
        "top_k": top_k,
        "category_filter": category,
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
            "BM25, cosine-similarity, and reciprocal-rank-fusion scores use "
            "different scales. Compare ranking positions and reviewed relevance, "
            "not the raw numeric scores across methods."
        ),
        "filter_note": (
            "The category filter is applied after retrieving each method's top-50 "
            "candidate pool; results are not a full-corpus category-filtered search."
            if category
            else None
        ),
        "evidence_note": (
            "Historical results contain ticket descriptions, not verified "
            "resolution steps."
        ),
    }
