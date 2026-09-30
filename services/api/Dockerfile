# =============================================================================
# 케어노트AI — API (FastAPI + LangGraph) 이미지
#
# [What] 멀티스테이지 빌드: builder(의존성 설치) → runtime(실행물만).
# [Why]  빌드 도구를 최종 이미지에서 제거해 크기·공격면을 줄이고,
#        non-root 유저로 실행해 컨테이너 탈취 피해를 최소화한다.
# [How]  현업 파이썬 서비스 표준 패턴. 채워야 할 곳은 TODO 로 표시.
# =============================================================================

# ---- 1) builder: 의존성 설치 --------------------------------------------------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# TODO: 의존성 매니페스트를 먼저 복사해 레이어 캐시를 살린다.
#       pyproject.toml 을 채운 뒤 아래 중 프로젝트 방식에 맞는 한 줄을 택한다.
COPY pyproject.toml ./
# (A) pip + pyproject:
#   RUN pip install --prefix=/install .
# (B) requirements.txt 방식이면:
#   COPY requirements.txt ./ && RUN pip install --prefix=/install -r requirements.txt
RUN echo "TODO: install deps into /install" && mkdir -p /install

# ---- 2) runtime: 실행에 필요한 것만 -------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/install/bin:$PATH" \
    PYTHONPATH="/install/lib/python3.12/site-packages"

# non-root 유저 생성 (uid 고정 → 볼륨 권한 예측 가능)
RUN groupadd -r app && useradd -r -g app -u 10001 app

WORKDIR /app

# builder 가 설치한 패키지만 가져온다 (빌드 도구는 버림 → 이미지 최소화)
COPY --from=builder /install /install

# 애플리케이션 소스
COPY app ./app

USER app

EXPOSE 8000

# 헬스체크: GET /health → 200 이어야 healthy
# TODO: app/main.py 에 @app.get("/health") 로 {"status":"ok"} 추가
HEALTHCHECK --interval=15s --timeout=5s --retries=5 --start-period=20s \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# 로그는 stdout 으로(uvicorn 기본). 프로덕션은 --workers 조정.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
