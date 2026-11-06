# IT-Sentinel — Intelligent IT Ticket Analysis & Resolution Assistant

IT-Sentinel is an API-first analyst-assistance prototype. FastAPI owns the ML and retrieval workflow; Streamlit is a separate analyst-facing client that calls the API. It classifies incoming IT tickets, retrieves similar historical ticket descriptions, searches explicitly approved knowledge, captures analyst feedback, and keeps an SQLite audit trail.

It is decision support only. It does **not** close tickets, change access, execute scripts, or perform privileged remediation.

## 1. Architecture

```text
Analyst
  |
  v
Streamlit dashboard (frontend/app.py)
  | HTTP/JSON
  v
FastAPI (src/api/main.py)
  |-- Classification: calibrated classifier (src/classification)
  |-- Priority: demonstration impact/urgency matrix
  |-- BM25 retrieval: lexical search
  |-- Semantic retrieval: Sentence Transformer embeddings
  |-- Hybrid retrieval: Reciprocal Rank Fusion
  |-- Recommendation: approved knowledge base only
  |-- Feedback + audit: SQLite (data/it_sentinel.db)
  v
Processed historical ticket descriptions + model/index artifacts
```

## 2. Current capabilities

- Ticket category prediction with top category probabilities and a human-review flag.
- P1–P4 demonstration priority estimate when impact and urgency are supplied.
- Similar historical ticket retrieval using BM25, semantic search and hybrid RRF.
- Side-by-side retrieval comparison endpoint.
- Resolution recommendations only when matching articles are explicitly marked approved and contain source attribution.
- Analyst feedback (`useful`, `incorrect`, `unresolved`).
- Request-level and ticket-level audit history.
- Streamlit UI for all the above.

## 3. Requirements and expected artifacts

Python 3.11 is the tested project environment. The API expects these existing project artifacts:

```text
models/ticket_classifier.joblib
models/ticket_embeddings.npy
models/ticket_embedding_metadata.csv
models/embedding_config.json   # if used by your local indexing module
 data/processed/classification_train.csv
```

The historical ticket corpus contains 1,570 indexed training ticket descriptions. Internal `ITS-TRAIN-...` IDs are project-generated references, not original production ticket numbers.

Install local dependencies from the project root:

```bash
source .venv/bin/activate
python -m pip install -r requirements-api.txt
python -m pip install -r requirements-dashboard.txt
```

If the project already has a complete `requirements.txt`, keep it and add the dashboard dependencies rather than replacing it. `rank-bm25` is required for the BM25 retrieval module.

## 4. Run locally on macOS

Use two VS Code terminals.

**Terminal A — API**

```bash
cd "/Users/debarghamishra/Intelligent it ticket analysis"
source .venv/bin/activate
python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8001
```

The first startup loads the classifier, BM25 corpus, embeddings and embedding model. Wait for `Application startup complete`.

**Terminal B — dashboard**

```bash
cd "/Users/debarghamishra/Intelligent it ticket analysis"
source .venv/bin/activate
python -m pip install -r requirements-dashboard.txt
IT_SENTINEL_API_URL=http://127.0.0.1:8001 python -m streamlit run frontend/app.py --server.port 8501
```

Open:

- Dashboard: http://localhost:8501
- API health: http://127.0.0.1:8001/health
- Swagger: http://127.0.0.1:8001/docs

If port 8001 is busy, stop the earlier Uvicorn process with Control+C in its terminal, or select another unused port and update `IT_SENTINEL_API_URL` in the dashboard terminal.

## 5. Dashboard areas

1. **Overview** — API/model status, system components and operating limitations.
2. **Analyze ticket** — category, confidence, top category probabilities and demonstration priority.
3. **Similar tickets** — hybrid search with category filter and ticket evidence views.
4. **Compare retrieval** — BM25, semantic and hybrid results side by side.
5. **Recommendations** — approved knowledge matches, source references and historical ticket descriptions.
6. **Feedback & audit** — save analyst feedback and inspect ticket/request audit records.

## 6. API routes

| Method | Route | Purpose |
|---|---|---|
| GET | `/health` | API/model readiness |
| POST | `/api/v1/tickets/analyze` | Ticket classification + demo priority |
| POST | `/api/v1/tickets/search` | Hybrid similar-ticket retrieval |
| POST | `/api/v1/tickets/search/compare` | BM25/semantic/hybrid comparison |
| POST | `/api/v1/tickets/recommend` | Evidence-backed recommendation lookup |
| POST | `/api/v1/feedback` | Save analyst feedback |
| GET | `/api/v1/audit/tickets/{ticket_id}` | Ticket audit history |
| GET | `/api/v1/audit/requests/{request_id}` | One request and feedback history |

## 7. Approved knowledge articles

`data/knowledge_base/approved_articles.json` must be a JSON list. Only articles with `approval_status: "approved"` are indexed. Every approved article must contain:

- `article_id`, `title`, `content`, `resolution_steps`
- `approval_status`
- `source_name`, `source_reference`
- `approved_by`, `approved_at`

Do not populate this file with invented fixes. Use internal runbooks, vendor documentation or other sources your organization has permission to use, and record approval metadata. The starter file is empty, so recommendation requests correctly fall back to human review until approved articles are loaded.

Example structure (replace the placeholders only with a real, reviewed source):

```json
[
  {
    "article_id": "KB-EXAMPLE-001",
    "title": "Reviewed article title",
    "content": "A concise description from an authorized knowledge source.",
    "resolution_steps": ["Approved step one", "Approved step two"],
    "approval_status": "approved",
    "source_name": "Authorized internal knowledge base",
    "source_reference": "https://your-authorized-source/article/KB-EXAMPLE-001",
    "approved_by": "Approver role or authorized identifier",
    "approved_at": "2026-11-06"
  }
]
```

After updating the JSON, restart the API because the knowledge base is loaded during application startup.

## 8. Automated tests

Install the test dependencies used in the existing suite, then run:

```bash
python -m pip install pytest httpx
python -m pytest tests -v
```

The previously run API suite contained 14 passing tests. Those tests validate API behaviour with test fixtures; they do not establish model accuracy or retrieval relevance. Re-run the suite after copying this dashboard bundle and after any API changes.

## 9. Docker (optional)

Docker Desktop must be installed and running. The Docker build needs the complete `src/`, `data/` and `models/` directories in the project root.

```bash
docker compose up --build
```

Open:

- Dashboard: http://localhost:8501
- API docs: http://localhost:8000/docs
- API health: http://localhost:8000/health

The API container persists SQLite data through the `./data:/app/data` mount. Model artifacts are mounted read-only. The first container startup may take longer while model resources initialize and any required model files are downloaded.

## 10. Evaluation status and limitations

- The classification test split results previously observed were accuracy 0.7976, macro F1 0.7537 and weighted F1 0.7903 on 657 examples. These are dataset-specific results, not a guarantee on operational tickets.
- Retrieval diagnostics used 18 selected queries and AI-assisted provisional relevance labels (93 provisional relevant, 59 provisional irrelevant, 39 unreviewed). These are not human-verified ground truth. Do not report them as final benchmark metrics.
- The current historical ticket corpus exposes descriptions, not verified agent resolutions. Similar tickets are evidence of precedent only; they must not be rendered as proven fixes.
- The priority matrix is a project demonstration rule, not an official HCLTech SLA or production priority policy.
- Classification confidence is a model probability estimate, not a guarantee of correctness.
- Category filtering is applied to a retrieved candidate pool, not necessarily to the full corpus before ranking; see API `filter_note`.
- The current API has no authentication, role-based authorization, rate limiting, tenant isolation or production secrets management. Keep it on localhost or a trusted private development network. Do not expose it publicly with real ticket data until those controls and organizational privacy/security reviews are implemented.
- Audit data may contain ticket text and analyst notes. Treat `data/it_sentinel.db` as sensitive; do not commit it to source control or upload it publicly.

## 11. Before production deployment

1. Replace demo priority rules with an approved service-management policy.
2. Review the retrieval benchmark manually and create a held-out, human-judged test set.
3. Load only approved, attributable knowledge articles.
4. Add API authentication, authorization, rate limiting and secure secrets.
5. Apply data retention, redaction, access-control and audit policies.
6. Add monitoring, backup/restore and deployment-specific health checks.
7. Re-test the complete system against real but authorized/de-identified workflows.

**Project status:** local prototype/dashboard-ready, not production-approved.
