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

## 명령어 (스캐폴딩 후 확정)
```bash
uv run pytest        # 테스트
uv run ruff check .  # 린트
uv run pyright       # 타입 검사
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
- 대회 공고 미공개: 제출 형태·배점 확정 시 blueprint 5·9장 조정

## 현재 단계
- 지금: 설계 (blueprint v1.1 (1인 기준, /court 반영))
- 다음: ① 스캐폴딩 ② 골든셋 스키마 + 20건 ③ Eval Runner 뼈대 (2주차에 E2E 데모 v0)
