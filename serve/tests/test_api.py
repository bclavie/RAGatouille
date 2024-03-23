from fastapi.testclient import TestClient
from .payloads import index_documents, search_query, add_documents, delete_documents, rerank, encode, search_encoded, clear_encoded
from ragatouille_serve.main import app

client = TestClient(app)

def test_index():
    response = client.post("/api/v1/index", json=index_documents)
    assert response.status_code == 200

def test_search():
    response = client.post("/api/v1/search/", json=search_query)
    assert response.status_code == 200

def test_add_to_index():
    response = client.post("/api/v1/documents/add", json=add_documents)
    assert response.status_code == 201
    assert response.json() == {"message": "Documents added to index successfully"}

def test_delete_from_index():
    response = client.delete("/api/v1/documents/delete", params=delete_documents)
    assert response.status_code == 200
    assert response.json() == {"message": "Documents deleted from index successfully"}

def test_rerank():
    response = client.post("/api/v1/rerank/", json=rerank)
    assert response.status_code == 200

def test_encode():
    response = client.post("/api/v1/encode/", json=encode)
    assert response.status_code == 200
    assert response.json() == {"message": "Documents encoded successfully"}

def test_search_encoded_docs():
    response = client.post("/api/v1/search-encoded/", json=search_encoded)
    assert response.status_code == 200

def test_clear_encoded_docs():
    response = client.post("/api/v1/clear-encoded/clear", json=clear_encoded)
    assert response.status_code == 200
    assert response.json() == {"message": "Encoded documents cleared successfully"}