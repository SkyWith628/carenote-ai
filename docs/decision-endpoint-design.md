# 사람 결정 엔드포인트 설계 (blueprint 3·5장 "확정은 사람")

구현: `src/carenote/drafts/` (`models.py` · `service.py` · `store.py` · `api.py` · `static/review.html`), 테스트 `tests/test_decisions.py`(23개, API 호출 없음).

## 흐름
`POST /drafts` → `in_review`(revision 1, author=ai) → 검토자가 값을 고치거나 확인 → `POST /drafts/{id}/decision` → `approved`(수정이 있으면 revision 2, author=user) 또는 `rejected`. 한 번 결정되면 끝이다.

## 요청
```json
{"decision": "approve", "edits": {"pulse": "72"}, "confirmed": ["blood_pressure", "fall_event"], "reason": null}
```
- `edits`: 필드 키 → 새 값(공백 제거, 1~200자). `confirmed`: 값을 그대로 받아들이는 필드. `reason`: 거절 사유(선택, 500자). 정의되지 않은 필드·키는 422(`extra=forbid`).

## 규칙
| 상황 | 결과 |
|---|---|
| approve인데 flag가 ok가 아닌 필드가 `edits`·`confirmed` 어디에도 없음 | **422** + `{"message", "unresolved": [키…]}`. 아무것도 바뀌지 않음 |
| approve, 값 수정 있음 | 새 revision(author=user, `diff: {키: {from, to}}`), 수정한 필드는 `edited=true`, `flag=ok`, `source=stated`, 근거·후보·검사 결과는 비움 |
| approve, 수정 없이 확인만 | revision 유지, 값 그대로 |
| 현재 값과 같은 `edits` | 변경으로 세지 않음(revision 미증가, `edited` 아님) |
| reject | 필드·revision 그대로, 상태만 `rejected`. 미해결 필드가 있어도 가능 |
| 이미 approved/rejected | **409**. 동시 요청 8개 중 정확히 1개만 성공(저장소 락 안에서 원자적 갱신) |
| 없는 초안 | 404 |

핵심 안전 규칙은 첫 줄이다. `확인 필요`·`위험`을 사람이 보지 않고는 승인할 수 없다. `위험` 필드(예: 미복용)는 오류가 아니라 사실일 수 있으므로 "확인함"으로 받아들일 수 있다.

## 기록 (append-only)
- `revisions`: AI 초안(rev 1)은 수정되지 않고 보존된다. 사람 수정은 새 revision으로 쌓인다(blueprint 5장 `draft_revisions`).
- `audit`: `created`(system), `approve`/`reject`(actor). approve 상세에는 **필드 키만**(`edited=[pulse] confirmed=[…]`) 남기고 값·원문·거절 사유 내용은 남기지 않는다(로그 본문 금지 원칙). 거절은 "사유 있음" 여부만 기록한다.

## 검토 화면
필드마다 "수정할 값" 입력과(확인 필요·위험 필드에는) "확인함" 체크박스, 하단에 승인·거절 버튼과 거절 사유 입력. 미해결 항목이 있으면 항목 이름을 알려준다. 거절은 확인 대화상자를 거친다. 결정 뒤에는 입력을 없애고 "승인됨 (revision 2)"를 보여준다. 가짜 추출기를 붙인 임시 서버에서 브라우저로 승인 차단→수정·확인→승인, 거절→409 재시도를 직접 확인했다.

## 한계 / 아직 없음
- **인증이 없다.** 행위자는 고정값 `reviewer`다. 인증·역할(RBAC) 전에는 외부에 열면 안 된다.
- **저장은 인메모리**다(재시작하면 사라짐). Postgres·RLS·영속 audit_log는 아직이다.
- 사람이 쓴 값은 evidence 없이 `source=stated`로 기록한다("모든 필드는 evidence 필수" 원칙의 예외). `source=human` 추가를 검토할 수 있으나 골든셋 스키마(`Source`)에 영향이 있어 보류했다.
- 수정 값의 형식 검증이 없다(예: 혈압 `130/80` 모양). 수치 필드는 Flagger 범위 규칙을 수정 값에도 적용할지 결정이 필요하다.
- 값을 비우는(missing으로 되돌리는) 수정과 승인 후 재수정(재개)은 지원하지 않는다.
- 모바일 레이아웃은 CSS 미디어쿼리만 있고 실기기 확인은 하지 않았다.
