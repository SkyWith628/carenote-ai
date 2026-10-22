"""FastAPI 진입점."""

from fastapi import FastAPI

from carenote.config import get_settings


def create_app() -> FastAPI:
    """FastAPI 앱을 생성한다."""
    settings = get_settings()
    app = FastAPI(title=settings.app_name)

    @app.get("/health")
    async def health() -> dict[str, str]:
        """헬스체크."""
        return {"status": "ok"}

    return app


app = create_app()
