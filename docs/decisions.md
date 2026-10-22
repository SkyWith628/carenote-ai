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
