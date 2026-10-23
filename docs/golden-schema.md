# 골든셋 스키마

코드: `src/carenote/evaluation/schema.py`, 로더: `loader.py`, 데이터: `data/golden/*.jsonl (현재 v1.jsonl, 20건)` (한 줄에 케이스 1건).

| 항목 | 규칙 |
|---|---|
| id | `gs-NNN`, 전체에서 유일 |
| category | normal · missing · ambiguous_number · dialect · stt_error · risk_combo · injection |
| synthetic | 항상 `true`. 실제 수급자 데이터 금지 |
| transcript | 마스킹된 합성 문장. 주민번호·휴대전화 패턴이 있으면 거부 |
| reference_transcript | stt_error 케이스에서만 사용 (실제 발화) |
| injection_attempt | injection 케이스에서만 `true` |
| fields | 8개 필드 전부. 빠지거나 중복되면 거부 |

필드 정답: `key, value, source(stated/inferred/missing), evidence[], flag(ok/needs_confirm/risk), candidates[]`

일관성 규칙
- `missing`: value·evidence·candidates 모두 비어야 한다. 위험 필드면 flag는 `needs_confirm`.
- 값이 있으면 evidence가 필요하고, 각 evidence 텍스트는 transcript에 그대로 들어 있어야 한다.
- 모호한 값: `value=null`, `candidates` 2개 이상, `needs_confirm`.

## 현재 구성 (v1, 20건)
| 유형 | 건수 | id |
|---|---|---|
| normal | 3 | 001, 005, 006 |
| missing | 3 | 002, 007, 008 |
| ambiguous_number | 3 | 003, 009, 010 |
| dialect (경상·전라·충청) | 3 | 011~013 |
| stt_error | 3 | 014~016 |
| risk_combo | 3 | 017~019 |
| injection | 2 | 004, 020 |

- stt_error는 `transcript`가 오인식 결과, `reference_transcript`가 실제 발화이다. 정답값은 reference 기준이고, 텍스트만 보는 시스템의 최선은 `needs_confirm`이다 (Re-listener 효과를 재는 케이스).
- risk flag 기준 수치(예: 160/100)는 임시값이다. Rule Engine YAML에서 확정하고 골든셋을 맞춘다.

## 다음
- Eval Runner: 필드 F1, 근거 없는 값 비율, 위험 recall 계산
- 60건 목표까지 유형별 확장 (3주차 이후)
