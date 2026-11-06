from fastapi import HTTPException, Request
from typing import Any, Dict


def get_resources(request: Request) -> Dict[str, Any]:
    resources = getattr(request.app.state, "resources", None)
    if not resources or not resources.get("ready"):
        raise HTTPException(status_code=503, detail="IT-Sentinel models are not ready.")
    return resources
