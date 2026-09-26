import httpx
from fastapi import FastAPI

from booking_truth.serve import BackgroundServer


def test_background_server_serves_and_stops() -> None:
    app = FastAPI()

    @app.get("/ping")
    def ping() -> dict[str, str]:
        return {"pong": "yes"}

    with BackgroundServer(app) as server:
        assert server.port > 0
        response = httpx.get(f"{server.url}/ping", timeout=5)
        assert response.json() == {"pong": "yes"}
