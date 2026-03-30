from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict():
    payload = {
        "transaction": {
            "TransactionID": 1,
            "TransactionAmt": 100.0,
            "ProductCD": "W"
        },
        "identity": None
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "ensemble_score" in response.json()
    assert "prediction" in response.json()
    assert response.json()["prediction"] in [0, 1]
