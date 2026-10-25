# 신규 케이스 스펙 (골든셋 v2)

정답을 먼저 쓰고 transcript는 나중에 생성한다(`docs/golden-expansion-design.md` 5장). 이 폴더는 **transcript가 없는 스펙**이다. 아직 모델에 노출되지 않았다.

파일: `probe_<프로브>.json` (H 환각 압력, C 정정·복수 값, D 방해 수치·제3자, N 부정과 시점, L 길고 지저분한 발화, S STT 오류(숫자 외), V 어휘 경계, G 방언·구어. 이후 I R 추가 예정)

| 키 | 의미 |
|---|---|
| `spec_id`, `probe`, `split` | 식별자, 프로브 코드, `dev`/`test` (test는 최종 보고 때만 실행) |
| `scene` | 생성기에게 줄 상황 설명 |
| `style` | 화자, 지역, 격식, 말더듬 정도 (케이스마다 다르게) |
| `length_chars` | transcript 길이 목표 [최소, 최대] |
| `must_include` / `must_not` | 반드시 들어갈 사실 / 들어가면 안 되는 단서 |
| `traps` / `hypothesis` | 유도하는 실패와 사전 가설 (결과를 보기 전에 기록) |
| `policy` | 적용되는 `docs/label-policy.md` 항목 |
| `label_risk` | 사람 라벨러가 갈릴 위험(low/medium/high). medium 이상은 독립 라벨링 표본에 우선 포함 |
| `gray_fields` | 정책상 정답이 갈릴 수 있는 필드(주로 `special_notes`). 라벨러·모델이 달라도 오답으로 단정하지 않고 별도 집계하기 위한 표시. 러너 반영은 구현 단계에서 결정 |
| `stt` | S 프로브 전용. `errors[]`에 대상 필드, 참고 발화 `reference`, 오인식 결과 `heard`, 종류, **텍스트만으로 복원 가능한지**(`recoverable_from_text`)를 기록. false인 필드는 텍스트 전용 모델에게 구조적 오답이므로 모델 오류와 분리해 집계한다. S 케이스의 정답은 reference 기준 |
| `gold` | 8개 필드의 정답 `{value, source, flag, candidates}` (evidence는 생성 후 transcript에서 인용해 채움) |

가설은 결과를 보기 전의 예측이다. 틀렸다고 지우지 않고 결과와 함께 비교한다.
