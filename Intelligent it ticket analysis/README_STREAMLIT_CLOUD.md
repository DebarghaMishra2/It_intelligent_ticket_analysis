# IT-Sentinel — Streamlit Community Cloud profile

This deployment profile serves the existing Streamlit UI directly from the
repository. It does not require a separately hosted FastAPI service.

## Important differences from the full local/Docker build

- Reuses the existing scikit-learn ticket classifier.
- Reuses the existing BM25 index builder and corpus.
- Uses TF-IDF cosine similarity as a lightweight second retrieval signal.
- Uses Reciprocal Rank Fusion over BM25 + TF-IDF.
- Does **not** load PyTorch or Sentence Transformers. The UI labels this method
  as TF-IDF, not neural semantic search.
- Approved knowledge articles are read from
  `data/knowledge_base/approved_articles.json`. Only articles marked approved
  and containing resolution steps can produce a recommendation.
- Feedback and audit are stored in SQLite at `/tmp/it_sentinel_streamlit.db` by
  default. Streamlit Community Cloud local storage is ephemeral; records can
  be lost after app restart, sleep, or redeployment. Do not rely on this for
  permanent audit or real organisational tickets.

The original FastAPI backend and Sentence Transformer retrieval modules remain
in the repository for local development and a higher-memory deployment.

## Install into the existing project

Extract this patch, then run from the project root:

```bash
python scripts/install_streamlit_cloud.py "/absolute/path/to/Intelligent it ticket analysis"
```

Run the installer from the project virtual environment (`source .venv/bin/activate`). It backs up files it replaces and pins the Streamlit runtime dependencies to versions already installed in that environment. It does not modify `frontend/workbench.py` or your FastAPI backend.

## GitHub / Streamlit Cloud

1. Commit and push the updated `frontend/`, `.streamlit/config.toml`, and root
   `requirements.txt`.
2. In Streamlit Community Cloud, create an app from the repository's `main`
   branch and set the main file path to `frontend/app.py`.
3. In Advanced settings → Secrets, add:

```toml
IT_SENTINEL_MODE = "local"
```

4. Deploy. Streamlit Cloud installs root `requirements.txt`.

## Required committed runtime assets

Verify these exist in the GitHub `main` branch:

- `models/ticket_classifier.joblib`
- `data/processed/classification_train.csv`
- `data/knowledge_base/approved_articles.json`
- `src/classification/predict.py`
- `src/preprocessing/priority_rules.py`
- `src/retrieval/bm25_search.py`
- `src/retrieval/indexing.py` and its required preprocessing modules

The lightweight profile uses `build_bm25_index()` from the existing BM25 module.

## Local testing

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
IT_SENTINEL_MODE=local python -m streamlit run frontend/app.py
```

## Operational limitations

- The second retrieval signal is TF-IDF lexical similarity, not transformer-based
  semantic embeddings. Do not describe it as neural semantic search in the demo.
- Ticket priority remains a demonstration impact/urgency matrix, not an official
  HCLTech SLA.
- The current dataset contains historical ticket descriptions, not verified
  agent resolutions. Recommendations remain in human-review fallback unless
  approved knowledge articles with resolution steps are added.
- Streamlit Cloud filesystem and SQLite records are temporary. Use an external
  managed database before relying on persistent feedback/audit or handling
  confidential tickets.
- Add authentication/authorization and data access controls before production use.
