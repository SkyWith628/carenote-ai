"""헬스체크 엔드포인트 테스트."""

from fastapi.testclient import TestClient

from carenote.main import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    """/health는 200과 status=ok를 반환한다."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_unknown_route_returns_404() -> None:
    """없는 경로는 404다."""
    assert client.get("/nope").status_code == 404
