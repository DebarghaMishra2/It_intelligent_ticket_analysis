"""Local preflight check for the IT-Sentinel prototype."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = [
    "src/api/main.py",
    "src/api/services.py",
    "src/api/routes/tickets.py",
    "src/api/routes/feedback.py",
    "src/api/routes/audit.py",
    "src/classification/predict.py",
    "src/retrieval/bm25_search.py",
    "src/retrieval/semantic_search.py",
    "src/retrieval/hybrid_search.py",
    "models/ticket_classifier.joblib",
    "models/ticket_embeddings.npy",
    "models/ticket_embedding_metadata.csv",
    "data/processed/classification_train.csv",
    "data/knowledge_base/approved_articles.json",
    "frontend/app.py",
    "frontend/api_client.py",
]
REQUIRED_MODULES = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "numpy",
    "pandas",
    "sklearn",
    "joblib",
    "sentence_transformers",
    "rank_bm25",
    "streamlit",
    "requests",
]


def main() -> int:
    missing_files = [item for item in REQUIRED_FILES if not (ROOT / item).is_file()]
    missing_modules = [item for item in REQUIRED_MODULES if importlib.util.find_spec(item) is None]

    print("IT-SENTINEL PROJECT PREFLIGHT")
    print("=" * 54)
    print(f"Project root: {ROOT}")
    print("\nRequired files:")
    if missing_files:
        for item in missing_files:
            print(f"  MISSING  {item}")
    else:
        print(f"  OK       {len(REQUIRED_FILES)} required files found")

    print("\nPython modules:")
    if missing_modules:
        for item in missing_modules:
            print(f"  MISSING  {item}")
    else:
        print(f"  OK       {len(REQUIRED_MODULES)} required modules importable")

    print("\nTest suite:")
    test_dir = ROOT / "tests"
    test_files = sorted(test_dir.glob("test_*.py")) if test_dir.exists() else []
    print(f"  {len(test_files)} test modules found")

    if missing_files or missing_modules:
        print("\nPREFLIGHT FAILED: resolve missing items above.")
        return 1

    print("\nPREFLIGHT PASSED: required local project files and modules are present.")
    print("This check does not validate model quality, approved knowledge content, or production security.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
