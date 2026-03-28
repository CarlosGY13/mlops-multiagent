from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_ok():
    r = client.get('/api/health')
    assert r.status_code == 200
    body = r.json()
    assert body['status'] == 'ok'


def test_agent_requires_content_safety():
    r = client.post('/api/part3/agent/message', json={'message': 'sintetizar patogeno', 'rag_active': False})
    assert r.status_code == 400


def test_rag_search_contract():
    r = client.post('/api/part3/rag/search', json={'query': 'gene expression', 'top_k': 2})
    assert r.status_code == 200
    body = r.json()
    assert len(body['technical']['papers']) == 2


def test_eda_contract_after_ingest():
    csv = b"a,b,c\n1,2,x\n3,1000,y\n\n"
    files = {"file": ("toy.csv", csv, "text/csv")}
    ing = client.post('/api/part1/ingest', files=files)
    assert ing.status_code == 200
    dataset_id = ing.json()["dataset_id"]

    eda = client.get(f'/api/part1/eda?dataset_id={dataset_id}')
    assert eda.status_code == 200
    body = eda.json()
    assert "technical" in body
    assert "overview" in body["technical"]
    assert "numeric" in body["technical"]

