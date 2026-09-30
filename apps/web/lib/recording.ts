// 녹음 업로드/조회 API 클라이언트
//
// [What] 녹음 오디오를 백엔드로 올리고, 전사·초안 상태를 폴링한다.
// [Why]  화면(page.tsx)에서 네트워크 로직을 분리. lib/api.ts 와 같은 계층.
// [How]
//   uploadRecording(recipientId, blob) -> { recordingId }   (POST /api/recordings)
//   getRecording(recordingId) -> { status, transcript?, draftId? }  (GET /api/recordings/{id})
//        status: transcribing → transcribed → 초안(pending_review) 링크
//
// TODO: 학습하며 함께 구현
