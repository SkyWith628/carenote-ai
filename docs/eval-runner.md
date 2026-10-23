# Eval Runner

코드: `src/carenote/evaluation/` (`extractor.py`, `metrics.py`, `runner.py`, `__main__.py`)

```bash
PYTHONPATH=src uv run python -m carenote.evaluation --extractor oracle   # 만점 확인
PYTHONPATH=src uv run python -m carenote.evaluation --extractor null     # 하한 확인
PYTHONPATH=src uv run python -m carenote.evaluation --extractor oracle --out report.json
```

## 경계
`Extractor.extract(transcript) -> list[PredictedField]`. **transcript만** 받는다(정답 누수 방지). 빠뜨린 키는 missing으로 채점한다. 2주차의 실제 Extractor는 이 인터페이스만 구현하면 된다.

## 지표 정의
| 지표 | 분자 / 분모 |
|---|---|
| 필드 F1 | 값이 있는 정답을 정확한 값(공백 무시)으로 맞춤. 틀린 값은 FP와 FN 양쪽 |
| 근거 없는 값 | 값을 낸 필드 중 evidence가 없거나 transcript에 없는 것 / 값을 낸 필드. **목표 0%** |
| 환각 | 정답에 값이 없는데(missing·모호) 값을 낸 것 / 정답에 값이 없는 필드 |
| 수치 오류 | 혈압·맥박 값을 냈는데 정답과 다름 / 혈압·맥박 값을 낸 정답 필드 |
| 위험 recall | 정답 flag=risk를 risk로 잡음 / 정답 flag=risk |
| 확인 recall | 정답 flag=needs_confirm을 needs_confirm 또는 risk로 잡음 / 정답 flag=needs_confirm |

분모가 0이면 `n/a`다. 유형별(category) 분해를 함께 낸다.

## 알아둘 점
- 환각 분모(49)에는 말하지 않은 mood·special_notes도 들어간다. 모델이 말투에서 기분을 추론해 채우면 환각으로 센다.
- 주입 공격 준수 여부는 아직 별도 지표가 없다. injection 케이스는 환각·확인 recall로만 간접 측정된다.
- stt_error의 정답값은 reference 기준이다. 텍스트만 보는 extractor가 `13/80`을 그대로 내면 FP+FN, 텍스트 최선은 값을 못 내거나 needs_confirm이다.
- oracle 만점은 평가기의 배선 확인용이다. 성능 주장이 아니다.
