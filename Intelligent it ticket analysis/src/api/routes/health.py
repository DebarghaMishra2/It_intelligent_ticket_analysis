from fastapi import APIRouter, Request

router = APIRouter(tags=["Health"])


@router.get("/health")
def health(request: Request):
    resources = getattr(request.app.state, "resources", None)
    ready = bool(resources and resources.get("ready"))
    return {
        "status": "ok" if ready else "starting",
        "service": "IT-Sentinel API",
        "models_loaded": ready,
    }
