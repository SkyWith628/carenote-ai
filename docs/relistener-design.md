# Re-listener 텍스트 폴백 설계 (blueprint 4.1 · 9장 "오디오 모델 API 변동" 완화)

구현: `src/carenote/extraction/relistener.py`, 테스트 `tests/test_relistener.py`(가짜 클라이언트, API 호출 없음). **평가 미실행.**

## 역할
오디오 네이티브 Re-listener가 없을 때, 위험 필드(수치·복약·낙상)를 **주장을 모른 채로** 원문에서 다시 읽는다. Verifier는 "이 값이 맞나"를 물어 확인 편향이 생길 수 있는데, 이 폴백은 항목과 답의 형식만 주고 값을 읽게 한 뒤 **코드가** Extractor의 값과 비교한다.

## 판정 (FieldVerdict 재사용 → Consensus 그대로)
| 읽기 결과 | 판정 | 효과 |
|---|---|---|
| Extractor 값과 같고 인용이 실재(수치는 숫자가 인용에서 읽힘) | supported | flag 불변 |
| 근거 있게 읽었는데 값이 다름 | unsupported | ok → needs_confirm |
| conflict, 값 비움, 인용 없음·가짜, 값 형식 불일치 | ambiguous | ok → needs_confirm |
| 호출 실패·응답 불가 | unavailable | flag 불변, 기록만 |
flag는 올리기만 한다(기존 원칙).

## 설계 결정
- 같은 `FieldVerdict`/`apply_verdicts`/`FieldChecker` 프로토콜을 써서 `VerifyingExtractor(inner, relistener, name="relistener")`로 Verifier 위에 겹쳐 쌓는다.
- 캐시 키에 Extractor의 값을 넣지 않는다(읽기는 값과 무관). 같은 읽기를 다른 주장과 재비교해도 재호출이 없다.
- 모델은 Verifier와 같은 Haiku 5.5 effort low를 가정(설정 분리는 필요 시). Extractor(Sonnet)와 다른 티어다. 독립성은 "주장을 모르는 읽기"이지 다른 제공사가 아니다.

## 한계 / 미정
- 텍스트를 다시 읽는 것이라 **STT가 정보를 지운 경우(gs-014·015)는 못 잡는다**. 그건 오디오 재청취의 몫이다(오디오 어댑터가 같은 `verify` 모양을 구현하게 한다).
- Verifier와 같은 계열·같은 transcript라 오류가 상관될 수 있다. 추가 포착이 있는지는 dev에서 재야 한다.
- 비용: 위험 필드당 호출 1회가 Verifier와 별도로 더 든다. 운영 켜기 전에 ablation 행(+Re-listener 폴백)으로 이득을 보고 결정한다.
- 러너 `--relistener` 연결, dry-run 추정, LangGraph 노드(verify와 병렬 또는 verify 뒤)는 아직 없다.
