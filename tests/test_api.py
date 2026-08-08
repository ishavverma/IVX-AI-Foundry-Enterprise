"""Focused tests for the versioned Flask API."""

from app import create_app
from engines.dataset.connectors import DatasetConnector


def test_api_requires_json_authentication() -> None:
    client = create_app().test_client()
    response = client.get("/api/v1/datasets")
    assert response.status_code == 401
    assert response.get_json() == {"error": "authentication_required"}


def test_connector_source_resolution() -> None:
    github = DatasetConnector.github("acme", "corpus", "data/train.jsonl", "v1")
    assert github.uri.endswith("/v1/data/train.jsonl")
    assert github.filename == "train.jsonl"

    hf = DatasetConnector.huggingface("acme/corpus", "data/train.jsonl")
    assert hf.uri.endswith("/resolve/main/data/train.jsonl")
    assert hf.filename == "train.jsonl"

    kaggle = DatasetConnector.kaggle("acme/corpus", "train.csv")
    assert kaggle.uri.endswith("?filename=train.csv")
