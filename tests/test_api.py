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
