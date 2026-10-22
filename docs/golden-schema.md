# 골든셋 스키마

코드: `src/carenote/evaluation/schema.py`, 로더: `loader.py`, 데이터: `data/golden/*.jsonl` (한 줄에 케이스 1건).

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

## 다음
- 시드 4건(normal, missing, ambiguous_number, injection) → 20건으로 확장 (dialect, stt_error, risk_combo 추가)
- Eval Runner: 필드 F1, 근거 없는 값 비율, 위험 recall 계산
