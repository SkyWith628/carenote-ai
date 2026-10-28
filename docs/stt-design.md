# STT 설계: 서버 Whisper (blueprint 열린 결정 1)

구현: `src/carenote/stt/`(엔진·`POST /transcribe`), `src/carenote/privacy.py`(마스킹), `src/carenote/evaluation/{stt_metrics,stt_eval,tts_mac}.py`, 검토 화면 녹음 버튼. 테스트 `tests/test_stt.py`·`tests/test_stt_eval.py`(가짜 엔진·가짜 `say`).
**상태: 코드·테스트 완료, 실제 Whisper는 한 번도 돌리지 않음.** 이 Windows PC는 앱 제어 정책이 서명 없는 네이티브 모듈을 막아 faster-whisper(ctranslate2)를 로드할 수 없을 가능성이 높고, 모델도 내려받지 않았다. Mac에서 아래 "Mac 실행 순서"로 확인해야 한다.

## 결정과 근거 (2026-10-28, 사용자 선택)
| 선택지 | 판단 |
|---|---|
| **서버 Whisper(자체 호스팅) — 채택** | 오디오가 우리 서버를 벗어나지 않음. 합성 음성으로 한국어 정확도를 직접 재서 ablation에 넣을 수 있음. 오디오를 두면 Re-listener(오디오 재청취)로 확장 가능 |
| 브라우저 Web Speech API | 키·비용 0이지만 오디오가 브라우저 벤더 서버로 감(Chrome 기본). `processLocally`는 Chrome Android 미지원([MDN](https://developer.mozilla.org/docs/Web/API/SpeechRecognition/processLocally), [caniuse](https://caniuse.com/mdn-api_speechrecognition_processlocally)). 오디오 파일이 없어 고정 음성 WER·Re-listener 불가 |
| 외부 STT API | 새 키·종량 비용, 건강정보 음성이 제3자로 전송 |
- Anthropic API로 음성 처리는 확인되지 않았다(공식 문서에서 오디오 입력을 찾지 못함, 서드파티는 미지원으로 기재). 그래서 blueprint의 "오디오 네이티브 Re-listener"는 지금 전제가 없다. 오디오 Re-listener를 하려면 별도 오디오 모델이 필요하며, 지금은 텍스트 폴백(`docs/relistener-design.md`)이 있다.
- 한국어 Whisper 정확도의 신뢰할 만한 공개 비교는 찾지 못했다. 우리 합성 음성으로 직접 재야 한다.

## 흐름
`녹음(MediaRecorder) → POST /transcribe → 서버 STT(메모리 처리) → PII 마스킹 → 화면의 입력창 → 사람이 틀린 부분(특히 숫자) 수정 → POST /drafts(다시 마스킹) → 추출·검증 → 사람 확정`
- **2단계로 나눈 이유**: STT 오류를 LLM이 믿고 전파하는 것이 A(교차검증)의 대상 문제다. 전사문을 검토자가 먼저 고치게 하면 가장 값싼 방어선이 하나 더 생긴다. 단, 고친 글이 "원본 음성과 같은 글"이 아니므로 이 경로의 정확도는 STT 단독 정확도가 아니다.
- 오디오와 초안은 아직 연결되지 않는다(Re-listener 오디오 버전을 하려면 동의·단기 보관·삭제가 필요하다, blueprint 6장).

## `POST /transcribe`
요청 본문이 오디오(`Content-Type: audio/webm|ogg|mp4|mpeg|wav|aiff|flac…`, 코덱 파라미터는 무시). 응답 `{text, duration_s, language, engine, elapsed_s, masked_count}`.
| 상황 | 상태 |
|---|---|
| 형식 아님 | 415 |
| 4MB 초과(Content-Length 또는 스트림 누적, 청크 전송도 막음) | 413 |
| 빈 본문 | 400 |
| 음성 없음·손상·120초 초과 | 422 (메시지에 오디오·전사 내용 없음) |
| 엔진 불가(패키지·모델 없음) | 503 |
| 다른 변환 대기 60초 초과 | 503 + `Retry-After` |
모든 응답에 `Cache-Control: no-store`. 오디오는 `BytesIO`로만 다뤄 디스크에 쓰지 않고, 본문·전사는 로그에 남기지 않는다.

## 엔진 (`FasterWhisperEngine`)
기본 `large-v3-turbo`, `int8`, 언어 `ko`(설정 `stt_*`). 첫 요청에서 모델을 불러온다(다운로드 약 1.5GB 안팎, 첫 요청이 느림). 모델은 한 번에 한 요청만 쓴다(CPU 데모 기준). 환각 억제: VAD 필터 켬, `condition_on_previous_text` 끔, 길이는 디코딩 전에 `info.duration`으로 거른다. 패키지는 선택 의존성이라 `uv add --optional stt faster-whisper`가 필요하다(이 저장소의 `pyproject.toml`·`uv.lock`은 아직 안 바꿨다).

## 개인정보 마스킹 (`privacy.py`)
STT 이후, LLM·저장소 이전에 주민등록번호·휴대전화·장기요양인정번호 **형식**을 치환한다(`/transcribe` 출력, `POST /drafts` 입력 모두). 감사 기록에는 횟수만 남긴다. **한계**: 한글 수사로 읽은 번호("공일공 일이삼사…")와 이름은 잡지 못한다. 이 단계는 "개인정보가 없다"는 근거가 아니다.

## 평가 도구 (Mac)
1. `tts_mac`: dev 케이스 transcript를 `say -v Yuna`로 합성(test 분할은 거부 — STT 튜닝이 test를 보게 되므로).
2. `stt_eval`: 음성 → STT → **CER**(공백·문장부호 무시)과 **수치 보존율**. CER은 "130에 80"에서 숫자 하나가 틀려도 작아 보이므로 수치 보존율을 따로 본다(테스트로 이 둘의 차이를 고정했다). 수치 보존율은 낙관적 상한이다(`extract_numbers`가 한글 수사를 여러 방식으로 읽음).
3. 다음 단계(미구현): 받아 적은 글(`--hyp-out`)을 추출기에 넣어 "음성→STT→추출"의 최종 필드 정확도를 텍스트 입력과 비교 — 이것이 STT 오류가 A 구성(Verifier·Re-listener)에서 얼마나 잡히는지 보여 주는 실제 ablation이다.

## Mac 실행 순서
```bash
uv add --optional stt faster-whisper
python -m carenote.evaluation.tts_mac --list-voices           # 한국어 음성 확인 (Yuna 등)
python -m carenote.evaluation.tts_mac --out data/golden/audio  # dev 30건, 파일은 .gitignore 필요
python -m carenote.evaluation.stt_eval --audio-dir data/golden/audio --out docs/reports/stt-dev.json --hyp-out /tmp/hyp.json
uv run uvicorn carenote.main:app --app-dir src --reload        # /review에서 녹음 시험
```
첫 `stt_eval`은 모델 다운로드로 오래 걸린다. `data/golden/audio`는 용량이 커서 커밋하지 않는다.

## 한계 / 미정
- **실측 없음**: 정확도(CER·수치)와 지연(RTF)은 Mac에서 처음 재야 한다. 데모 노트북 CPU에서 `large-v3-turbo`가 몇 초 걸릴지 모른다(느리면 `small`·`medium`과 비교).
- **합성 음성은 낙관적**이다(깨끗한 음질, 방언·잡음·겹침 없음). 실제 정확도로 말하면 안 된다. 직접 녹음 소량(blueprint 7장)이 필요하다. 실제 수급자 음성은 쓰지 않는다.
- 인증이 없다. `/transcribe`는 외부에 열면 안 된다(요청 크기·동시성 제한은 있으나 호출 제한(rate limit)은 없음).
- 모바일 사파리(mp4)·안드로이드 크롬(webm) 녹음은 이 PC의 브라우저에서 오실레이터 스트림으로만 확인했고 실기기·실제 마이크는 확인하지 않았다.
- 서버 STT 지연(수 초)이 방문당 약 5초 추출 지연(`docs/cost-latency.md`)에 더해진다. 전체 end-to-end 지연은 아직 측정하지 않았다.

## 첫 실측 (2026-10-28, Mac M3 8GB, CPU int8, `large-v3-turbo`, 합성 음성 dev 30건)
보고서 `docs/reports/20261028-stt-dev-large-v3-turbo.json`. 음성은 macOS `say`(Yuna)로 만든 깨끗한 음질이라 **낙관적 상한**이다.

| 항목 | 값 |
|---|---|
| CER | micro 4.5% · 건별 평균 6.1% (방언 14.4%, 모호한 수사 10.8%, STT 오류 유형 7.8%) |
| 지연 RTF | 평균 1.9 · 건별 중앙 2.13 (1.08~3.92). 음성 19.5초 평균에 받아쓰기 37초(최대 80.8초) |
| 메모리 | 최대 RSS 약 0.9GB |
| 모델 다운로드 | 약 1.6GB, 약 10분 (불안정한 연결) |

- **지연: 이 장비에서는 라이브 시연에 느리다.** 실시간보다 약 2배 느리다. 같은 시간에 테스트·린트를 돌려 약간 부풀었을 수 있으나 방향은 바뀌지 않는다. `small`·`medium` 비교와 시연 장비 실측이 필요하다.
- **조용한 수치 오류(핵심 발견)**: 활력징후 정답값 48개 중 38개(79%)만 인식문에 그대로 있었다. 틀린 것은 대부분 끝자리 "이"가 사라진 패턴이다: 72→70(gs-001·041·055), 82→80(gs-016·028), 혈압 130/82→130/80. "칠십이예요"의 "이"를 서술격 조사(이에요·이었어요)로 읽은 것으로 보이나 원인은 확정하지 못했다. 틀린 값이 그럴듯해서 사람이 못 알아챈다. 또 gs-013 "120에 80이구유"→"120에 82구유"(방언).
- **지표 결함**: `numbers_preserved`는 87/120(72.5%)로 나왔지만 **기준문 쪽 수사를 여러 방식으로 읽어 기대 집합이 부풀려진 인공물**이다(예: "백삼십 팔십이었나"에서 100·103·104 등). 실제 손실을 과대 표시하므로 이 값을 보고에 쓰지 않는다. 위의 활력징후 정답값 대조(골든 라벨 기준)를 쓴다. 지표는 라벨 값 기준으로 고쳐야 한다.
- **의미**: 텍스트 Verifier·Re-listener는 이 오류를 못 잡는다(STT가 이미 지운 정보). 오디오를 다시 듣는 Re-listener가 필요한 근거이고, 수치 필드는 사람 확인을 거치게 하는 안전망의 필요를 뒷받침한다.
- 한계: 합성 음성 30건, 1회, 한 장비. 실제 방문 음성의 잡음·방언·겹침은 더 어렵다. 이 결과로 "STT 정확도"를 말하지 않는다.
