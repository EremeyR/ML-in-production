from fastapi.testclient import TestClient

from inference import app
import pandas as pd

import sys
sys.path.append('..')

from common_artifacts.utils import create_features

client = TestClient(app)


def test_health_without_startup():
    response = client.get("/health")
    assert response.status_code == 400
    # assert not response.json()["state"]


def test_health_with_startup():
    client.put("/startup")
    response = client.get("/health")
    assert response.status_code == 200



def test_one_line_inference():
    client.put("startup")

    test_features = create_features(1)
    request_features = list(test_features.columns)

    response = client.get(
        "/predict",
        json={"data": [test_features.iloc[0].tolist()], "features": request_features}
    )
    assert response.status_code == 200
    assert response.json()[0]["label"] > -1
    assert response.json()[0]["label"] < 2


def test_double_one_line_inference():
    client.put("startup")

    test_features = create_features(2)
    request_features = list(test_features.columns)
    for i in range(2):
        response = client.get(
            "/predict",
            json={"data": [test_features.iloc[i].tolist()], "features": request_features}
        )
        assert response.status_code == 200
        assert response.json()[0]["label"] > -1
        assert response.json()[0]["label"] < 2


