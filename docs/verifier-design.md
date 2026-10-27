# Verifier 설계 (blueprint 4.1, 4주차)

구현: `src/carenote/extraction/verifier.py`, 테스트 `tests/test_verifier.py`(가짜 클라이언트, API 호출 없음).

## 역할
위험 필드(수치·복약·낙상, `RISK_FIELDS`) 중 **값이 있는 필드**를 transcript와 다시 대조한다. KR-1(번복이 풀리지 않은 혈압을 확정)과 만들어진 값을 잡는 두 번째 눈이다. 값이 없거나 모호한 필드는 이미 확인 필요라 건너뛴다.

## 설계 결정
| 결정 | 선택 | 이유 |
|---|---|---|
| 모델 계열 | Claude 내 다른 티어: Extractor Sonnet 5.5, Verifier **Haiku 5.5** (effort low) | 열린 결정 3. API 키가 Anthropic 하나뿐이고 비용이 낮다($0.10/$0.50 per MTok). **한계**: 같은 제공사라 오류가 상관될 수 있다. "타 제공사" 독립성은 주장하지 않는다. 발표에는 "다른 티어"로 쓴다. |
| 입력 | transcript 전체 + 주장(항목, 값). **Extractor의 근거 인용은 주지 않는다** | 앵커링 방지. 번복 판단(KR-1)에는 전문이 필요하다. blueprint의 "필드 + 근거 구간"에서 벗어난 점이라 ablation으로 비교할 후보(근거 주기)로 남긴다. |
| 호출 단위 | 필드당 1회 | blueprint 8.1 비용 틀(위험 필드 수 = 호출 수)과 일치. 판정 격리. |
| 출력 | `verdict`(supported / unsupported / ambiguous) + `quote`(결정 구간) | |
| 코드 검사 | V1 supported의 인용이 transcript에 실재해야 함. V2 수치 필드는 값의 숫자가 인용에서 읽혀야 함(`extract_numbers`). 어기면 `ambiguous`로 강등 | LLM은 제안, 코드가 판정(프로젝트 원칙). 강등은 flag를 올리는 방향이라 안전하다. |
| Consensus | unsupported·ambiguous면 `ok → needs_confirm`. risk·needs_confirm은 유지, **올리기만** 함 | 기존 Flagger·Sentinel 규칙과 같은 방향. |
| 실패 | `unavailable`로 기록, flag 불변 | 실패를 "문제없음"으로 읽지 않도록 verdict에 남긴다. 운영에서는 UI에 "검증 못 함"을 표시해야 한다(남은 일). |
| 인젝션 | transcript는 `<transcript>` 자료로 감싸고 지시를 따르지 않도록 명시, 출력은 스키마로만 | blueprint 6장 |

## 알려진 한계
- 인용의 실재와 숫자 일치는 증명하지만, 인용이 판정을 **의미상** 뒷받침하는지는 증명하지 못한다.
- supported인데 틀린 경우(Verifier가 Extractor와 같은 실수)는 못 잡는다. ablation에서 그 비율을 보고한다.
- 복약·낙상은 숫자 검사(V2)가 없어 인용 실재만 본다.

## 평가 계획 (아직 실행 안 함)
- dev 분할만 사용. test는 이미 `--final-report` 1회를 썼으므로 이 단계의 튜닝에 쓰지 않는다.
- ablation 행 추가: 단일 → +근거 강제(B1) → **+Verifier**. 지표: 근거 없는 값 비율, 수치 오류율, 위험 recall, 오경보(정답인데 needs_confirm으로 올린 비율), 비용·지연.
- 가장 먼저 볼 것: gs-003(KR-1)에서 Flagger 규칙만 vs +Verifier.
- 실행 전 dry-run으로 비용을 추정하고 승인액을 받는다(Sentinel LLM 실행과 같은 절차).

## 남은 일
1. 평가 러너 연결(`--verifier` 옵션, dry-run, 비용 가드) 후 dev 실행.
2. (완료 2026-10-28) `/drafts` API 연결: `get_extractor`가 `VerifyingExtractor`로 감싸고, 필드 응답에 `checks.verifier`(supported·unsupported·ambiguous·unavailable), 검토 화면에 "재검증" 줄을 표시한다. 검증 실패는 초안 생성을 막지 않고 "검증하지 못함"으로 보인다.
3. 근거 인용을 주는 변형과의 비교.
