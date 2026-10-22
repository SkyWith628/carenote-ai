"""환경 설정. 비밀값은 .env에서만 읽고 커밋하지 않는다."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """애플리케이션 설정."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="CARENOTE_", extra="ignore")

    app_name: str = "carenote"
    environment: str = "local"


def get_settings() -> Settings:
    """설정 인스턴스를 반환한다."""
    return Settings()
