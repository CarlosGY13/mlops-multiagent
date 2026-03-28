# Lab Notebook AI (v1.1)

Backend FastAPI + UI web con tabs por etapas (Ingesta, Entrenamiento, Drift/Produccion, Agente) y panel lateral on-demand para razonamiento cientifico sobre experimentos no hardcodeados.

## Ejecutar

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## UI

Open `http://127.0.0.1:8000/` to view the staged flow and switch Researcher/Technical view.

## Partes

- Parte 1: `/api/part1/*` ingesta + calidad
- Parte 2: `/api/part2/*` entrenamiento + gates + drift
- Parte 3: `/api/part3/*` agente + RAG on-demand + panel lateral

`USE_LOCAL_MOCK=true` por defecto para desarrollo sin Azure real.

### Live RAG (Europe PMC + OpenML)
Set `USE_LOCAL_MOCK=false` to use real HTTP calls for the related research side panel.
