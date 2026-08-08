"""Fast platform smoke tests suitable for CI."""

from app import create_app


def test_public_operational_endpoints() -> None:
    app = create_app()
    client = app.test_client()
    assert client.get("/health").status_code == 200
    assert client.get("/metrics").status_code == 200
    assert client.get("/api/openapi.json").status_code == 200


def test_anonymous_access_and_csrf() -> None:
    app = create_app()
    client = app.test_client()
    assert client.get("/").status_code in {302, 303}
    assert client.post("/login", data={"username": "x", "password": "x"}).status_code == 400
