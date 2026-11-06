from fastapi import APIRouter, HTTPException

from src.api.database import get_audit_by_request_id, get_audit_by_ticket_id

router = APIRouter(prefix="/api/v1/audit", tags=["Audit"])


@router.get("/tickets/{ticket_id}")
def audit_for_ticket(ticket_id: str):
    records = get_audit_by_ticket_id(ticket_id)
    return {"ticket_id": ticket_id, "record_count": len(records), "records": records}


@router.get("/requests/{request_id}")
def audit_for_request(request_id: str):
    record = get_audit_by_request_id(request_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Audit record not found.")
    return record
