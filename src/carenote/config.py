"""환경 설정. 비밀값은 .env에서만 읽고 커밋하지 않는다."""

from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """애플리케이션 설정."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="CARENOTE_", extra="ignore")

    app_name: str = "carenote"
    environment: str = "local"

    # Anthropic SDK의 표준 환경변수 이름을 그대로 쓴다 (CARENOTE_ 접두사 없음).
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    extractor_model: str = "claude-sonnet-5-5"
    extractor_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    extraction_cache_dir: str = ".cache/extractions"


def get_settings() -> Settings:
    """설정 인스턴스를 반환한다."""
    return Settings()
