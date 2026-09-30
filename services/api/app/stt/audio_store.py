"""오디오 단기보관 — 저장·조회·TTL 파기.

[What] 업로드된 녹음 오디오 원본을 짧은 기간만 보관하고 자동 파기한다.
[Why]  오디오는 재현·감사용 최소기간만 남긴다(개인정보 최소보관 원칙).
       전사(transcript)가 끝나면 원문 오디오는 곧 파기 대상.
[How]  오브젝트 스토리지(예: S3 호환)나 로컬 볼륨을 추상화. audio_ref(키)로
       저장/조회하고, expires_at 지난 항목은 배치로 하드 삭제(soft delete 선행).
       모든 접근은 감사로그(변조불가) 대상.
"""

# TODO: 학습하며 함께 구현
#   def put(data: bytes, *, mime: str) -> str:        # -> audio_ref
#   def get(audio_ref: str) -> bytes:
#   def purge_expired(now) -> int:                    # TTL 지난 오디오 파기
