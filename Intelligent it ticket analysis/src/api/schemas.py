from typing import Literal, Optional
from pydantic import BaseModel, Field


class TicketAnalyzeRequest(BaseModel):
    ticket_id: Optional[str] = Field(default=None, max_length=100)
    description: str = Field(..., min_length=5, max_length=20000)
    impact: Optional[Literal["low", "medium", "high"]] = None
    urgency: Optional[Literal["low", "medium", "high"]] = None


class TicketSearchRequest(BaseModel):
    ticket_id: Optional[str] = Field(default=None, max_length=100)
    query: str = Field(..., min_length=3, max_length=20000)
    top_k: int = Field(default=5, ge=1, le=10)
    category: Optional[str] = Field(default=None, min_length=1, max_length=100)


class TicketRecommendRequest(BaseModel):
    ticket_id: Optional[str] = Field(default=None, max_length=100)
    description: str = Field(..., min_length=5, max_length=20000)
    top_k: int = Field(default=5, ge=1, le=10)


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., min_length=1, max_length=80)
    feedback: Literal["useful", "incorrect", "unresolved"]
    ticket_id: Optional[str] = Field(default=None, max_length=100)
    notes: Optional[str] = Field(default=None, max_length=5000)
