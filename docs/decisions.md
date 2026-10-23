# 결정 로그

## 2026-10-22 — 1인 프로젝트로 범위 축소 (v1.1)
- 배경: /court 재판 유죄 5:1. 주요 사유는 팀 규모 미확정 대비 과다 범위, 공고 없는 "최소 수상 라인" 단정, 비어 있는 ablation 표.
- 결정: 팀은 1인. 피치를 "A(교차검증 멀티에이전트) + 라이브 데모"로 축소, B·C는 택1 stretch, 얇은 E2E 데모를 2주차로 당김, 기능 동결 6주차 말.
- 근거 기록: `.court/20261022-144415/VERDICT.md`
- 재검토 시점: 대회 공고 공개 시, 3주차 게이트

## 2026-10-23 — 골든셋 스키마 확정 (방문기록 1종)
- 결정: 케이스는 합성 transcript + **전 필드(8개) 정답**을 항상 나열한다. 값 없음은 `source=missing`으로 명시한다.
- 필드: visit_start, blood_pressure, pulse, meal_intake, medication_taken, fall_event, mood, special_notes. 위험 필드는 혈압·맥박·복약·낙상.
- 규칙: 위험 필드가 missing이면 정답 flag는 `needs_confirm`. 화자가 번복한 수치는 `value=None` + `candidates` 2개 이상 + `needs_confirm` (임의 선택은 오답, 되묻기가 정답).
- 근거: evidence는 transcript의 부분 문자열이어야 한다. 개인정보 패턴(주민번호·휴대전화)이 있으면 로드 단계에서 거부한다.
- 미정: 필드 목록은 열린 결정 3번(문서 타입·필수 항목)의 잠정안이다. 대회 서식이 나오면 조정한다.

## 2026-10-23 — 골든셋 v1 20건
- 유형별 3건씩(injection 2건). 방언은 경상·전라·충청, stt_error는 숫자 탈락·자릿수 오인식·낙상 키워드 변형.
- 주의: 20건은 스키마와 러너 검증용이다. 수치 주장은 60건 이후에 한다. 방언·말투는 작성자(1인)가 쓴 합성문이라 실제 구어체와 다를 수 있다.
- 위험 flag 수치 기준은 임시. Rule Engine 확정 시 재정렬.

## 2026-10-23 — Eval Runner 지표 정의
- 지표 6개(F1, 근거 없는 값, 환각, 수치 오류, 위험 recall, 확인 recall)와 분모 규칙은 `docs/eval-runner.md`가 원본이다.
- Extractor 경계는 `extract(transcript)`로 고정. 골든셋 정답은 extractor에 전달하지 않는다.
- 기준선: oracle(만점), null(하한 0)로 평가기 자체를 검증한다. ablation 표의 "단일 LLM" 행은 2주차 Extractor 완성 후 채운다.

## 2026-10-23 — Extractor 설계 (2주차)
- B0(단일 LLM)와 B1(+근거 강제) 두 변형으로 ablation 첫 두 행을 측정한다. 상세는 `docs/extractor-design.md`.
- flag는 LLM이 내지 않고 코드(Flagger)가 계산한다. 3주차 Rule Engine이 대체한다.
- `special_notes`는 v0에서 값 유무로 채점한다. 정확 일치는 불가능하다.
- 모델은 설정값, 기본 `claude-opus-5-5`로 품질 상한을 먼저 보고 Sonnet·Haiku 행을 추가한다 (사용자 확인 대기).
