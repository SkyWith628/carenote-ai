"""FastAPI 진입점."""

from fastapi import FastAPI

from carenote.config import get_settings
from carenote.drafts.api import router as drafts_router
from carenote.stt.api import router as stt_router


def create_app() -> FastAPI:
    """FastAPI 앱을 생성한다."""
    settings = get_settings()
    app = FastAPI(title=settings.app_name)

    @app.get("/health")
    async def health() -> dict[str, str]:
        """헬스체크."""
        return {"status": "ok"}

    app.include_router(drafts_router)
    app.include_router(stt_router)
    return app


app = create_app()
