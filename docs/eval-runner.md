# Eval Runner

코드: `src/carenote/evaluation/` (`extractor.py`, `metrics.py`, `runner.py`, `__main__.py`)

```bash
PYTHONPATH=src uv run python -m carenote.evaluation --extractor oracle   # 만점 확인
PYTHONPATH=src uv run python -m carenote.evaluation --extractor null     # 하한 확인
PYTHONPATH=src uv run python -m carenote.evaluation --extractor oracle --out report.json
```

## LLM 실측 CLI (비용이 든다)
```bash
# 1) 예상 비용만 보기. 네트워크·API 키 불필요
PYTHONPATH=src uv run python -m carenote.evaluation --extractor llm-b1 --dry-run
PYTHONPATH=src uv run python -m carenote.evaluation --extractor llm-b0 --model claude-haiku-5-5 --dry-run

# 2) 실행. .env에 ANTHROPIC_API_KEY가 있어야 하고, 승인할 최대 금액(USD)을 직접 지정한다
PYTHONPATH=src uv run python -m carenote.evaluation --extractor llm-b1 --approve-cost 1.00 --out b1.json

# 처음에는 소수 케이스로
PYTHONPATH=src uv run python -m carenote.evaluation --extractor llm-b1 --limit 3 --approve-cost 0.30
```
분할: `--split dev|test|all` (기본 **dev**: v1 20 + v2 10 = 30건). test(30건)는 최종 보고용이라 **LLM 실행은 `--final-report` 없이는 거부**한다(종료 코드 2). oracle·null은 제한 없음. test 실행 횟수는 `docs/decisions.md`에 기록한다.

프롬프트: `--prompt v1|v2|v3|v4` (기본 **v4**, 2026-10-26 기본 후보로 채택). v1~v3는 이전 보고서 재현용이다(v1 기준선, v2 정책 반영, v3 +기분 추측 허용). v4 설계는 `docs/extractor-prompt-v4-design.md`. 버전은 결과 캐시 키에 들어가 v1 캐시와 섞이지 않는다. v1 본문은 sha256으로 테스트가 고정한다.

반복 실행: `--run-tag T`는 같은 설정을 캐시 없이 다시 실행하는 표식이다(실행 간 변동 측정용). 태그가 없으면 캐시 키가 이전과 같아 기존 캐시가 적중한다. 보고서 JSON에는 케이스·필드별 결과 `per_field`(tp/fn/fpfn/fp/tn)가 들어가고 `usage.run_tag`가 기록된다.

변동 분석(`docs/run-variance-design.md`):
```bash
PYTHONPATH=src uv run python -m carenote.evaluation.variance r0.json r1.json r2.json r3.json r4.json
PYTHONPATH=src uv run python -m carenote.evaluation.variance X0.json X1.json ... --vs Y0.json Y1.json ...
```
σ_run, 쌍별 평균 |ΔF1|, 불안정 필드, 안정 오류, 다수결 F1을 낸다. `--vs --targets CASE:FIELD ...`를 더하면 목표 필드(예: `gs-018:meal_intake`)의 정답 횟수를 기준 묶음 대 새 묶음으로 세어 사전 기준(재현·개선 경향·회귀·유지·구분 불가, 피셔 단측 p)으로 판정한다. 프롬프트 v4부터 판정은 이 목표 필드 기준을 쓴다. `--vs`는 사전 규칙으로 판정한다: 평균 F1 차이가 σ_run 합성값의 2배보다 크고 케이스를 재표집한 실행 평균 결과의 95% 구간이 0을 포함하지 않을 때만 "차이 있음".

Ablation 옵션: `--prompt-edits 1,3,4,5`(`--prompt v4` 전용, v3에 v4의 해당 편집만 적용; 전부면 v4와 같고 캐시 키도 같다), `--case-ids gs-003,gs-012`(선택한 분할 안의 케이스만; test 케이스는 dev 분할에서 지정할 수 없다). 변형 × 케이스 × 반복을 한 비용 한도로 돌리는 도구는 `python -m carenote.evaluation.ablate`(`docs/v4-ablation-design.md`).

옵션: `--model`(기본 설정값 `claude-sonnet-5-5`), `--effort`(기본 `medium`), `--limit N`, `--cache-dir`, `--out`.

안전장치
- `--approve-cost`가 없으면 추정만 보여주고 **실행을 거부**한다(종료 코드 2). 키가 없거나 단가표에 없는 모델도 거부한다.
- 추정 최대 비용이 승인액을 넘으면 거부한다. 실행 중 누적 비용이 승인액에 도달하면 중단한다(종료 코드 3). 한도를 넘는 양은 최대 호출 1회분이다.
- 끝낸 호출은 `.cache/extractions/`에 저장되므로 승인액을 올려 다시 실행하면 이어서 진행된다. 캐시 적중은 비용이 0이다.
- 추정은 글자 수 기반이라 오차가 크다(입력은 비교적 맞고 출력은 추론량 때문에 5배 범위). 실제 비용은 보고서의 `usage`에 찍힌다. 단가는 2026-10-06 기준 캐시이므로 실측 전에 공식 가격을 다시 확인한다.
- LLM 실행 결과에는 항상 Flagger가 붙는다(`FlaggingExtractor`).

## 경계
`Extractor.extract(transcript) -> list[PredictedField]`. **transcript만** 받는다(정답 누수 방지). 빠뜨린 키는 missing으로 채점한다. 2주차의 실제 Extractor는 이 인터페이스만 구현하면 된다.

## 지표 정의
| 지표 | 분자 / 분모 |
|---|---|
| 필드 F1 | 값이 있는 정답을 정확한 값(공백 무시)으로 맞춤. 틀린 값은 FP와 FN 양쪽. **`special_notes`만 예외**: 자유 문장이라 값 유무로만 채점(표현이 달라도 둘 다 값이 있으면 일치, 내용의 옳고 그름은 보지 않음) |
| 근거 없는 값 | 값을 낸 필드 중 evidence가 없거나 transcript에 없는 것 / 값을 낸 필드. **목표 0%** |
| 환각 | 정답에 값이 없는데(missing·모호) 값을 낸 것 / 정답에 값이 없는 필드 |
| 수치 오류 | 혈압·맥박 값을 냈는데 정답과 다름 / 혈압·맥박 값을 낸 정답 필드 |
| 위험 recall | 정답 flag=risk를 risk로 잡음 / 정답 flag=risk |
| 확인 recall | 정답 flag=needs_confirm을 needs_confirm 또는 risk로 잡음 / 정답 flag=needs_confirm |

| 조용한 오답 | 위험 필드(혈압·맥박·복약·낙상)에서 값을 냈고 flag가 ok인데 정답과 다름 / 위험 필드에서 값을 내고 flag가 ok인 예측. 낮을수록 안전 |
| 과다 확인 | 정답 flag가 ok이고 값이 맞은 필드를 needs_confirm으로 올림 / 정답 flag가 ok이고 값이 맞은 필드. 틀린 값을 올린 것은 세지 않음 |
| 기권 | 값이 있는 정답 필드(special_notes 제외)에서 값을 아예 내지 않음 / 그 필드 수 |
| 오답 | 같은 분모에서 틀린 값을 냄 |

분모가 0이면 `n/a`다. 기권과 오답을 나눠 보는 이유: F1에서는 틀린 값이 FP+FN으로 기권보다 두 배 깎이지만, 사람에게는 기권(확인 요청)이 더 안전하다. 값을 내지 않는 쪽이 유리한 문제(gs-015)를 눈에 보이게 한다.

## 보고서 구성 (v2)
- 전체 · **복원가능 필드만**(`recoverable=false` 제외) · **복원가능·gray 제외**(`gray_fields`도 제외) 세 줄. 구조적 오답과 모델 오답을 분리한다.
- 유형별(category)과 프로브별(tags[0]; v1은 `v1`) 분해.
- **F1 95% 구간**: 케이스 단위 부트스트랩 1,000회, 시드 고정(필드가 아니라 케이스를 재표집: 한 케이스의 필드는 같은 transcript를 공유해 독립이 아니다).
- `--out` JSON에 `per_case`(케이스별 TP/FP/FN)가 들어간다.

## 구성 간 비교 (짝지은 차이)
```bash
PYTHONPATH=src uv run python -m carenote.evaluation.compare sonnet-b0.json sonnet-b1.json
# → "sonnet-b1 - sonnet-b0: +1.2%p [95% -0.4 ~ +3.0] → 구분 불가(구간이 0 포함)"
```
같은 케이스 집합이어야 한다(다르면 거부). 구간이 0을 포함하면 이 골든셋으로는 두 구성을 구분할 수 없다고 읽는다. `per_case`가 없는 이전 보고서는 현재 러너로 캐시 재채점해 다시 저장한다.

## 알아둘 점
- 환각 분모(49)에는 말하지 않은 mood·special_notes도 들어간다. 모델이 말투에서 기분을 추론해 채우면 환각으로 센다.
- 주입 공격 준수 여부는 아직 별도 지표가 없다. injection 케이스는 환각·확인 recall로만 간접 측정된다.
- stt_error의 정답값은 reference 기준이다. 텍스트만 보는 extractor가 `13/80`을 그대로 내면 FP+FN, 텍스트 최선은 값을 못 내거나 needs_confirm이다.
- oracle 만점은 평가기의 배선 확인용이다. 성능 주장이 아니다.
