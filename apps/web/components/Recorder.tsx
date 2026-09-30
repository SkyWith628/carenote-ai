// Recorder — 브라우저 MediaRecorder 캡슐화 컴포넌트
//
// [What] 마이크 권한 요청 → 녹음 시작/정지 → 녹음된 Blob(webm/opus) 반환.
// [Why]  녹음 로직을 화면에서 분리해 재사용·테스트 가능하게. 오디오는
//        브라우저 안에서만 생성되고, 업로드 시점에만 서버로 나간다.
// [How]  navigator.mediaDevices.getUserMedia → MediaRecorder. 녹음 시간·상태
//        (idle/recording/stopped) 표시. onComplete(blob, durationSec) 콜백.
//
// TODO: 학습하며 함께 구현
