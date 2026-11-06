"""FastAPI application entry point for IT-Sentinel."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.database import initialize_database
from src.api.services import initialize_resources
from src.api.routes import audit, feedback, health, tickets

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("it-sentinel")


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_database()
    logger.info("Loading IT-Sentinel classifier and retrieval indexes...")
    app.state.resources = initialize_resources()
    logger.info("IT-Sentinel API resources loaded.")
    yield
    app.state.resources = None
    logger.info("IT-Sentinel API shutdown complete.")


app = FastAPI(
    title="IT-Sentinel API",
    description=(
        "An analyst-assistance API for IT ticket classification, hybrid retrieval, "
        "evidence-backed knowledge recommendations, feedback, and audit history. "
        "It does not execute remediation or close tickets."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(tickets.router)
app.include_router(feedback.router)
app.include_router(audit.router)
