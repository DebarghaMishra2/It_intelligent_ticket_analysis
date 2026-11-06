from uuid import uuid4

from fastapi import APIRouter, Depends

from src.api.database import create_audit_record
from src.api.dependencies import get_resources
from src.api.schemas import (
    TicketAnalyzeRequest,
    TicketRecommendRequest,
    TicketSearchRequest,
)
from src.api.services import (
    build_recommendation,
    classify_and_prioritize,
    compare_retrieval_methods,
    search_similar_tickets,
)

router = APIRouter(prefix="/api/v1/tickets", tags=["Tickets"])


@router.post("/analyze")
def analyze_ticket(payload: TicketAnalyzeRequest, resources=Depends(get_resources)):
    request_id = str(uuid4())
    result = classify_and_prioritize(
        payload.description, resources, payload.impact, payload.urgency
    )
    response = {
        "request_id": request_id,
        "ticket_id": payload.ticket_id,
        **result,
    }
    create_audit_record(
        request_id, payload.ticket_id, "/api/v1/tickets/analyze",
        payload.description, response
    )
    return response


@router.post("/search")
def search_tickets(payload: TicketSearchRequest, resources=Depends(get_resources)):
    request_id = str(uuid4())
    results = search_similar_tickets(
        payload.query, resources, payload.top_k, payload.category
    )
    response = {
        "request_id": request_id,
        "ticket_id": payload.ticket_id,
        "query": payload.query,
        "category_filter": payload.category,
        "result_count": len(results),
        "results": results,
        "filter_note": (
            "Category filtering is applied to the fused top-50 candidate pool; "
            "it is not a full-corpus filtered search."
            if payload.category else None
        ),
    }
    create_audit_record(
        request_id, payload.ticket_id, "/api/v1/tickets/search",
        payload.query, response
    )
    return response


@router.post("/recommend")
def recommend_for_ticket(
    payload: TicketRecommendRequest, resources=Depends(get_resources)
):
    request_id = str(uuid4())
    result = build_recommendation(payload.description, resources, payload.top_k)
    response = {
        "request_id": request_id,
        "ticket_id": payload.ticket_id,
        "query": payload.description,
        **result,
    }
    create_audit_record(
        request_id, payload.ticket_id, "/api/v1/tickets/recommend",
        payload.description, response
    )
    return response


@router.post("/search/compare")
def compare_ticket_search_methods(
    payload: TicketSearchRequest, resources=Depends(get_resources)
):
    """Compare BM25, semantic, and hybrid retrieval for the same query."""
    request_id = str(uuid4())
    comparison = compare_retrieval_methods(
        payload.query,
        resources,
        top_k=payload.top_k,
        category=payload.category,
    )
    response = {
        "request_id": request_id,
        "ticket_id": payload.ticket_id,
        "query": payload.query,
        **comparison,
    }
    create_audit_record(
        request_id,
        payload.ticket_id,
        "/api/v1/tickets/search/compare",
        payload.query,
        response,
    )
    return response
