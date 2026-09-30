"""로컬 STT 래퍼 — 오디오 → (전사문, 신뢰도).

[What] 요양보호사 녹음 오디오를 텍스트로 전사한다.
[Why]  음성 원문엔 어르신 성함·주소 등 PII가 담겨 있어 외부 STT API로
       내보내면 안 된다 → 서버 내부 로컬/온디바이스 처리로 유출을 원천 차단.
       (마스터 기획서 §7 "마스킹 기본 ON" 정신)
[How]  faster-whisper 등 로컬 엔진을 감싸고, 전사문과 신뢰도를 함께 반환한다.
       신뢰도는 하류 룰 R008("음성 인식 불확실 → 원문 확인 필요")에서 사용.
       전사문은 이후 core/masking.py 를 통과한 뒤에만 LLM으로 전달된다.
"""

# TODO: 학습하며 함께 구현
#   def transcribe(audio_path: str, *, language: str = "ko") -> tuple[str, float]:
#       """오디오 파일 경로를 받아 (transcript, confidence) 반환."""
