# Lab Notebook AI (v1.1)

FastAPI backend + static web UI that guides non-technical researchers through:
- **Part 1**: dataset ingestion + schema inference + quality checks + quarantine
- **Part 2**: training/evaluation/deployment (Azure ML SDK v2)
- **Part 3**: scientific agent + on-demand related research search (RAG)

The UI supports **dual views** (Researcher vs Technical) at every stage.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open: `http://127.0.0.1:8000/`

## Environment variables

This repo reads env vars from **process env** and from a local **`.env` file**.

- `.env` is **gitignored** (never commit keys).
- For local development without Azure, keep:

```bash
USE_LOCAL_MOCK=true
```

### Azure ML (Part 2)

To run the **Azure ML pipeline** (feature engineering → train → evaluate → deploy):

```bash
USE_LOCAL_MOCK=false
AZURE_SUBSCRIPTION_ID="<your-subscription-id>"
AZURE_RESOURCE_GROUP="<your-resource-group>"
AZURE_ML_WORKSPACE_NAME="<your-ml-workspace-name>"
AZURE_REGION="<your-region>"   # e.g. centralus
AZURE_ML_COMPUTE_NAME="cpu-cluster"
AZURE_ML_COMPUTE_VM_SIZE="Standard_DS3_v2"
AZURE_ML_COMPUTE_MIN_NODES=0
AZURE_ML_COMPUTE_MAX_NODES=2
```

Notes:
- Authentication uses `DefaultAzureCredential`.
- The compute name must exist in the workspace (or update `AZURE_ML_COMPUTE_NAME`).

### Microsoft Foundry / OpenAI-compatible chat (Part 3)

You can configure the LLM using **either** OpenAI-compatible or Azure OpenAI-style settings.

**Option A — OpenAI-compatible**
```bash
FOUNDRY_OPENAI_BASE_URL="https://.../v1"
FOUNDRY_OPENAI_API_KEY="<key>"
FOUNDRY_OPENAI_MODEL="<model-name>"
```

**Option B — Azure OpenAI-style deployment**
```bash
FOUNDRY_AZURE_OPENAI_ENDPOINT="https://<resource>.cognitiveservices.azure.com"
FOUNDRY_AZURE_OPENAI_DEPLOYMENT="<deployment-name>"
FOUNDRY_AZURE_OPENAI_API_VERSION="2025-01-01-preview"  # if your endpoint requires it
FOUNDRY_OPENAI_API_KEY="<key>"  # same key field is used by the client
```

## API

- Part 1: `/api/part1/*` ingestion + quality + EDA
- Part 2: `/api/part2/*` ML pipeline (includes `/api/part2/aml/*`)
- Part 3: `/api/part3/*` agent + Search/RAG

## Live Search (Europe PMC + OpenML)

Set `USE_LOCAL_MOCK=false` to use real HTTP calls for the Search tab.
