#!/usr/bin/env python3
"""Pre-push checks for IT-Sentinel source and deployment artifacts."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "src/api/main.py", "src/api/database.py", "src/api/services.py",
    "src/classification/predict.py", "src/retrieval/bm25_search.py",
    "src/retrieval/semantic_search.py", "src/retrieval/hybrid_search.py",
    "frontend/app.py", "frontend/api_client.py",
    "data/processed/classification_train.csv",
    "data/knowledge_base/approved_articles.json",
    "models/ticket_classifier.joblib", "models/ticket_embeddings.npy",
    "models/ticket_embedding_metadata.csv",
    "Dockerfile.api", "Dockerfile.dashboard", "docker-compose.yml", "render.yaml",
]
FORBIDDEN_PARTS = {".venv", "venv", ".git", "__pycache__", ".pytest_cache"}


def main() -> int:
    print("IT-SENTINEL RELEASE PREFLIGHT")
    print("=" * 48)
    missing = [name for name in REQUIRED if not (ROOT / name).is_file()]
    for name in REQUIRED:
        print(("MISSING " if name in missing else "OK      ") + name)
    large = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or FORBIDDEN_PARTS.intersection(path.parts):
            continue
        if path.stat().st_size > 90 * 1024 * 1024:
            large.append((path.relative_to(ROOT), path.stat().st_size))
    if large:
        print("\nFiles larger than 90 MiB (GitHub ordinary file limit is 100 MiB):")
        for path, size in large:
            print(f"  {path}: {size / (1024 * 1024):.1f} MiB")
    print("\nSecurity reminder: keep .env, secrets, local databases, raw data and" )
    print("unapproved/private ticket data out of Git. Start with a private repository.")
    if missing or large:
        print("\nPREFLIGHT NEEDS ATTENTION")
        return 1
    print("\nPREFLIGHT PASSED (file presence/size only; not a security or Docker build test).")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
