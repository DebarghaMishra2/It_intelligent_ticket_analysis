from fastapi import APIRouter, HTTPException

from src.api.database import add_feedback
from src.api.schemas import FeedbackRequest

router = APIRouter(prefix="/api/v1/feedback", tags=["Analyst Feedback"])


@router.post("")
def submit_feedback(payload: FeedbackRequest):
    try:
        saved = add_feedback(
            payload.request_id,
            payload.ticket_id,
            payload.feedback,
            payload.notes,
        )
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return {"status": "saved", **saved}
