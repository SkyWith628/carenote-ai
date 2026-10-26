# carenote-ai
mode: work
> 요양보호사의 음성 방문 기록을 근거(evidence) 기반 구조화 초안으로 만들고, 교차검증 에이전트와 사람 승인을 거쳐 확정하는 AI 기록 도우미 (구현 경진대회, 2026-12 중순 예상).

## 목적 / 범위
- 팀: 개인 프로젝트(1인). 피치는 "A + 라이브 데모", B·C는 stretch 택1
- 목표: ① 골든셋에서 "근거 없는 값 0%" ② 단일 LLM 대비 환각·수치 오류 감소를 ablation 표로 증명 ③ 라이브 시연 가능한 end-to-end
- Non-goals: 실제 외부 시스템 연동(목업 폼만), 실제 수급자 데이터 사용(합성만), 결제·멀티 센터 운영 기능

## 스택 (구현 직전 공식 문서로 버전·모델 확정)
| 영역 | 기술 | 선택 이유 |
|---|---|---|
| API | Python 3.12 + FastAPI + uv | 주력 스택, 비동기 |
| 오케스트레이션 | LangGraph | 병렬 팬아웃·HITL interrupt·체크포인트가 실제로 필요 |
| DB | PostgreSQL 16 (JSONB, RLS) | 인프라 최소화, 테넌트 격리 |
| 웹 | Next.js + TS (PWA, 모바일 우선) | 현장 사용자는 폰 |
| 평가 | pytest + 자체 Eval Runner | 모든 혁신을 수치로 증명 |

## 명령어
```bash
uv run pytest        # 테스트
uv run ruff check .  # 린트
uv run pyright       # 타입 검사 (strict)
uv run python -m carenote.evaluation --extractor oracle   # 골든셋 평가 (PYTHONPATH=src 필요 시)
uv run uvicorn carenote.main:app --app-dir src --reload   # 개발 서버, GET /health
```

## 구조 (코드에서 안 보이는 것만)
- `docs/blueprint.md`: 설계도 원본 (이 문서가 설계의 진실)
- `docs/decisions.md`: 큰 결정 로그
- 이전 프로젝트: `../_archive/케어노트AI-v1` (시안 HTML 재사용 후보, 코드는 스켈레톤뿐)

## 컨벤션 · 결정
- LLM은 추출만, 판정은 코드 룰, 확정은 사람. 모든 필드는 evidence 필수, 없으면 `missing`
- 초안은 append-only revision. 하네스(프롬프트·룰) 변경은 골든셋 회귀 통과 + 승인 후 배포
- 테스트·데모 데이터는 합성만. 로그에 본문·식별정보 금지

## 함정 / 제약
- 음성은 STT 전에 마스킹 불가 → 온디바이스 STT 또는 센터 동의 설계 필요
- Harness2는 2차 보도만 확인(원문 미확인) → 의존 전 repo·논문 확인
- iCloud 경로에서는 `.venv`의 editable `.pth`에 hidden 플래그가 붙어 `import carenote`가 실패할 수 있다. pytest·pyright는 `src`를 직접 경로로 잡아 우회하고, 스크립트·uvicorn은 `--app-dir src` 또는 `PYTHONPATH=src`를 쓴다
- 대회 공고 미공개: 제출 형태·배점 확정 시 blueprint 5·9장 조정

## 현재 단계
- 지금: 1주차 완료분. 스캐폴딩, 골든셋 v1 20건, Eval Runner 뼈대 (`docs/eval-runner.md`)
- 지금 2주차: LLM Extractor B0/B1과 Flagger 구현 완료(가짜 클라이언트 테스트만, API 미호출), 설계 `docs/extractor-design.md`
- 1차 ablation 완료(`docs/ablation.md`): 20건은 포화돼 구성 간 차이가 노이즈. 다음: 골든셋 확장(설계 `docs/golden-expansion-design.md`, 열린 결정 5개 확인 필요)
- 골든셋 동결 도구 구현(`python -m carenote.evaluation.freeze --check|--write`, 아직 미실행). F2 미충족은 한계로 수용(불일치 하한 약 7%). 골든셋 v2 승격 완료(미동결): `data/golden/v2_dev.jsonl` 10건, `v2_test.jsonl` 30건, 평가 CLI 기본 분할 dev, test는 `--final-report` 필요. 동결 전 남은 일: R3·G1 현장 검토, 사람 독립 라벨링(95%), sha256·태그
- transcript 생성 도구 구현 완료(`docs/transcript-generation-design.md`, 파일럿은 `python -m carenote.generation --pilot --dry-run` 후 승인 시 실행)
- 실측 CLI(준비됨: `--dry-run`으로 비용 확인 → `--approve-cost`로 승인 후 실행)(`.env`에 ANTHROPIC_API_KEY 필요, 비용 승인 후) → `POST /extract` → 검토 화면. 원래 목표 — Extractor(LLM, 구조화 출력 + evidence)를 `Extractor` 인터페이스로 구현해 단일 LLM 베이스라인 측정, 검토 화면(원문 보기)
