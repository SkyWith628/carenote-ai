# LangGraph 오케스트레이션 설계 (blueprint 4장, 5주차 앞당김)

구현: `src/carenote/orchestration/graph.py`, 테스트 `tests/test_orchestration.py`.
**상태: 코드·테스트 작성 완료, 실행 미검증** (이 Windows PC의 앱 제어 정책이 `ormsgpack` DLL 로드를 차단해 langgraph를 import할 수 없음. Mac·USB 환경에서 `uv add langgraph` 후 확인 필요).

## 버전·API (공식 문서로 2026-10-28 확인)
langgraph 1.2.18 / langgraph-checkpoint 4.3.1. `StateGraph`, `Send`(조건부 엣지에서 팬아웃), `interrupt()` + `Command(resume=...)`(HITL, 같은 `thread_id`로 재개), `InMemorySaver`(운영은 영속 체크포인터로 교체). resume 시 interrupt가 있던 노드는 처음부터 다시 실행된다.

## 그래프
```
START → extract(Extractor + Flagger + Sentinel)
      → [Send verify × 값 있는 위험 필드]  (병렬; 대상이 없으면 consensus로 직행)
      → consensus(apply_verdicts, 코드)
      → human_review(interrupt: 필드·신호·verdict 전달)
      → finalize(approve/reject, 사람 수정 반영) → END
```
- LLM은 추출·재검증만, flag는 코드, 확정은 사람(프로젝트 원칙). 사람 결정이 없으면 그래프는 끝나지 않는다. `decision`이 approve가 아니면(누락·오타 포함) reject로 처리한다(기본값이 승인이면 안 된다).
- 상태는 JSON 호환 dict만 둔다(체크포인트 직렬화에서 pydantic·dataclass 복원 문제를 피하려는 선택). verdict는 `verdicts` 리듀서(키별 병합)로 병렬 결과를 합친다.
- 사람이 고친 값은 `source=stated`, `evidence=[]`, `flag=ok`로 기록한다(사람이 쓴 값이라 모델 근거가 없다). 계약의 "evidence 필수"에 대한 예외이며, 추후 `source=human` 추가를 검토한다.
- Verifier 실패는 `unavailable`로 verdicts에 남고 흐름은 계속된다.

## 아직 없는 것
- Re-listener 노드(텍스트 재검증 폴백부터), Clarifier(되묻기 → 해당 필드 재추출 루프), Sentinel 병렬화(지금은 extract 안에서 순차).
- `/drafts` API 연결(`POST /drafts`는 interrupt까지 실행, `POST /drafts/{id}/decision`으로 resume), 영속 체크포인터(Postgres).
- 병렬 이득의 지연 측정(8.1 표).

## 테스트가 확인하려는 것
위험 필드마다 정확히 1회 verify, 대상 없을 때 직행, 승인 전 미완료, 승인·거절·수정 반영, 알 수 없는 결정은 거절, 새 그래프 인스턴스가 같은 체크포인터로 재추출 없이 재개, 스레드 격리.
