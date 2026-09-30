// 음성 녹음 화면 (모바일 반응형) — 방문 현장에서 요양보호사가 사용
//
// [What] 수급자 선택 → 녹음 시작/정지 → 오디오 업로드 → 처리상태 표시.
// [Why]  현장에선 타이핑조차 부담 → "말로 하면 끝". 별도 앱 없이 휴대폰
//        브라우저에서 바로 쓰도록 기존 웹 SaaS 안에 라우트로 추가.
// [How]  <Recorder/>(MediaRecorder 캡슐화)로 녹음 → lib/recording.ts 로 업로드 →
//        상태 폴링(transcribing → transcribed → 초안 pending_review).
//        업로드 후 초안 생성·승인 큐 등재는 백엔드가 자동 처리("자동 전송").
//        최종 제출은 여전히 사회복지사가 /drafts/[id] 에서 수행(HITL).
//
// TODO: 학습하며 함께 구현
