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
    csv = b"x,y,target\n1,2,A\n4,5,A\n7,8,B\n"
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
    assert "features" in body["technical"]

    # bins param should change histogram resolution
    eda_bins = client.get(f'/api/part1/eda?dataset_id={dataset_id}&bins=8')
    assert eda_bins.status_code == 200
    b = eda_bins.json()
    hist = b["technical"]["numeric"]["x"]["hist"]
    assert len(hist["counts"]) == 8
    assert len(hist["bins"]) == 9

    eda2 = client.get(f'/api/part1/eda?dataset_id={dataset_id}&target_column=target')
    assert eda2.status_code == 200
    body2 = eda2.json()
    ta = body2["technical"]["target_analysis"]
    assert ta["task"] == "classification"
    assert ta["target"] == "target"
    assert len(ta["counts"]) >= 2


def test_curated_sample_contract():
    csv = b"a,b\n1,2\n3,4\n"
    files = {"file": ("toy.csv", csv, "text/csv")}
    ing = client.post('/api/part1/ingest', files=files)
    assert ing.status_code == 200
    dataset_id = ing.json()["dataset_id"]

    sample = client.get(f'/api/part1/curated/sample?dataset_id={dataset_id}&limit=2')
    assert sample.status_code == 200
    body = sample.json()
    assert body["technical"]["limit"] == 2
    assert len(body["technical"]["rows"]) == 2

