# IT-Sentinel: GitHub, Docker and Render deployment

This package adds a reproducible deployment layout for the existing project.
It keeps the API and Streamlit dashboard as separate containers and deploys them
as two Render web services from one Git repository.

## Important deployment assumptions

- Python 3.11.
- Runtime files needed by the API are committed: `models/`,
  `data/processed/classification_train.csv`, and
  `data/knowledge_base/approved_articles.json`.
- Raw ticket data, local SQLite files, reports, secrets and virtual environments
  are excluded from Git and Docker context.
- The embedding model is downloaded during the API image build and cached inside
  the image, rather than downloaded on every service restart.
- SQLite is still used. Render's API service writes its audit database to
  `/var/data/it_sentinel.db` on an attached persistent disk.
- The dashboard connects to the API over Render private networking using the
  `hostport` service reference. `frontend/api_client.py` is patched to prefix
  `http://` when a scheme is not supplied.

## 1. Install deployment configuration

Extract this package, then run from your Mac Terminal:

```bash
cd "/Users/debarghamishra/Intelligent it ticket analysis"
source .venv/bin/activate
python "$HOME/Downloads/it_sentinel_deploy_release/scripts/install_deploy_config.py" \
  "/Users/debarghamishra/Intelligent it ticket analysis"
```

The installer backs up files it changes. Inspect the generated backup files if
anything behaves differently than expected.

## 2. Verify local project assets

```bash
python scripts/preflight_check.py
python scripts/release_preflight.py
python -m pytest tests -v
```

The release preflight checks file presence and files larger than 90 MiB only; it
is not a security audit and does not execute Docker builds.

## 3. Test with Docker Compose on Mac

Install and open Docker Desktop first. Then:

```bash
docker --version
docker compose version
docker compose up --build
```

Open:
- API docs: <http://localhost:8000/docs>
- Dashboard: <http://localhost:8501>

The API data/database volume is a named Docker volume. `docker compose down`
keeps it. `docker compose down -v` deletes it; avoid `-v` if you want to retain
local feedback/audit records.

## 4. Prepare GitHub repository safely

Create a **new private, empty GitHub repository** for this project first. Do not
initialize it with a README, license or .gitignore because the local project
already has files. Keep it private until you have checked dataset licensing and
confirmed that all processed ticket text is approved for sharing.

In the project root:

```bash
git status --short
git remote -v
git branch --show-current
find . -type f -size +90M -not -path './.git/*' -not -path './.venv/*' -not -path './venv/*' -print
```

Review the output. Make sure `.env`, `.streamlit/secrets.toml`,
`data/it_sentinel.db`, `data/raw/`, `data/helpdesk/`, and any private ticket data
are not staged. Do not blindly stage a repository containing credentials.

If Git has not been initialized:

```bash
git init -b main
```

Stage and inspect before committing:

```bash
git add .
git status --short
git diff --cached --stat
git diff --cached --name-only
```

If you see secrets or raw/private data in the staged list, unstage them with
`git restore --staged PATH` and update `.gitignore` before continuing.

Commit:

```bash
git commit -m "Prepare IT-Sentinel for Docker deployment"
```

Connect the empty repository (replace the URL with the HTTPS URL copied from
GitHub):

```bash
# Use this if no origin exists:
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/it-sentinel.git

# If origin already exists and is the wrong repository, inspect it first,
# then use this instead of `remote add`:
# git remote set-url origin https://github.com/YOUR_GITHUB_USERNAME/it-sentinel.git

git branch -M main
git push -u origin main
```

GitHub may prompt you to authenticate with your browser or Git credential
manager. Never paste a personal access token into source code or commit it.

## 5. Deploy to Render with a Blueprint

1. Sign in to Render and connect your GitHub account.
2. In Render, choose **New + → Blueprint**.
3. Select the private IT-Sentinel repository and the `main` branch.
4. Render detects `render.yaml`; review the resources and costs before applying.
5. Apply the Blueprint. It creates `it-sentinel-api` and
   `it-sentinel-dashboard` in Singapore.
6. Wait for the API service to pass its `/health` check and for the dashboard to
   finish deploying.
7. Open the dashboard service's `onrender.com` URL. Open the API service's
   `/docs` URL to inspect Swagger.

The Blueprint configures the API on the `1c-2g` plan and attaches a 1 GB
persistent disk. Render documents that persistent disks are available only to
paid web/private services/workers. This is required here so SQLite audit and
feedback records survive restarts and redeploys. Review Render's current pricing
before approving. The dashboard is configured on the free plan and may spin down
when idle; upgrade it if you need an always-on demo.

The API and dashboard are in the same region. Render's `hostport` reference is
used for private service-to-service communication; the client adds `http://`.

## 6. Verify deployment

From your Mac, replace the placeholders with the URLs shown by Render:

```bash
curl https://YOUR-API-SERVICE.onrender.com/health
```

Open:

```text
https://YOUR-API-SERVICE.onrender.com/docs
https://YOUR-DASHBOARD-SERVICE.onrender.com
```

Submit a synthetic test ticket through the dashboard. Confirm that analysis,
search, recommendations, feedback and audit behave as expected. Do not use real
confidential tickets until authentication, authorization and data controls are
implemented.

## 7. Updates after deployment

After modifying code:

```bash
python -m pytest tests -v
git status --short
git add .
git diff --cached --stat
git commit -m "Describe the update"
git push origin main
```

Render's Git-connected services automatically deploy new commits by default.

## Known limitations / production blockers

- This project currently has no user authentication or role-based access control.
- The API exposes ticket/audit endpoints and must not receive confidential
  production tickets while publicly accessible without authentication.
- SQLite is suitable for this small single-instance prototype. For concurrent
  production usage or multiple API instances, migrate to a managed database.
- The priority matrix is demonstration logic, not an HCLTech-approved SLA.
- The current historical dataset has ticket descriptions, not verified agent
  resolution content. Keep recommendation fallback enabled until approved
  knowledge articles are added.
- The AI-generated retrieval labels are provisional, not final ground truth.
- Review dataset license and de-identification rules before changing the GitHub
  repository from private to public.
