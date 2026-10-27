"""STT 정확도 지표와 평가·합성 도구 테스트. 가짜 엔진·가짜 `say`만 쓴다."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from carenote.evaluation.loader import load_golden
from carenote.evaluation.stt_eval import evaluate, find_audio, main
from carenote.evaluation.stt_metrics import cer, edit_distance, normalize, numbers_preserved
from carenote.evaluation.tts_mac import main as tts_main
from carenote.evaluation.tts_mac import synthesize
from carenote.stt.engine import SttError, SttUnavailable, Transcription

GOLDEN = Path(__file__).parent.parent / "data" / "golden"

# -- 지표 ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b", "distance"),
    [("", "", 0), ("abc", "abc", 0), ("abc", "", 3), ("", "abc", 3), ("kitten", "sitting", 3)],
)
def test_edit_distance(a: str, b: str, distance: int) -> None:
    assert edit_distance(a, b) == distance and edit_distance(b, a) == distance


def test_normalize_ignores_spacing_and_punctuation() -> None:
    assert normalize("혈압은 130에 80이고요.") == normalize("혈압은130에80 이고요")


def test_cer_values() -> None:
    assert cer("혈압은 130에 80", "혈압은 130에 80").cer == 0.0
    result = cer("혈압은 130에 80", "혈압은 130에 90")
    assert (result.distance, result.ref_length) == (1, 9) and result.cer == pytest.approx(1 / 9)
    assert cer("", "").cer == 0.0 and cer("", "가").cer == 1.0


def test_numbers_preserved_counts_clinical_numbers() -> None:
    assert numbers_preserved("혈압은 130에 80", "혈압은 130에 80") == (2, 2)
    assert numbers_preserved("혈압은 130에 80", "혈압은 130에 90") == (1, 2)
    assert numbers_preserved("기분이 좋으세요", "기분이 좋아요") == (0, 0)
    assert (
        numbers_preserved("맥박은 칠십이고", "맥박은 72예요")[0] == 1
    )  # 한글 수사 ↔ 아라비아 숫자


def test_cer_hides_a_numeric_error_that_numbers_preserved_catches() -> None:
    reference = "어르신 오늘 점심 잘 드셨고 기분 좋으시고 혈압은 130에 80이었어요"
    hypothesis = "어르신 오늘 점심 잘 드셨고 기분 좋으시고 혈압은 130에 90이었어요"
    assert cer(reference, hypothesis).cer < 0.05  # 글자 오류는 작아 보이지만
    assert numbers_preserved(reference, hypothesis) == (1, 2)  # 숫자는 하나 틀렸다


# -- 평가 ----------------------------------------------------------------------------


class EchoEngine:
    """기준 transcript를 거의 그대로 돌려주는 가짜 엔진 (CER≈0). 마지막 글자를 바꿀 수 있다."""

    name = "echo"

    def __init__(self, texts: dict[bytes, str], fail: bytes | None = None) -> None:
        self.texts = texts
        self.fail = fail

    def transcribe(self, audio: bytes, content_type: str) -> Transcription:
        if audio == self.fail:
            raise SttError("음성이 감지되지 않았다")
        return Transcription(self.texts[audio], 3.0, "ko", self.name, 0.6)


def test_evaluate_scores_cases_and_reports_missing_and_failed(tmp_path: Path) -> None:
    (tmp_path / "a.wav").write_bytes(b"A")
    (tmp_path / "b.aiff").write_bytes(b"B")
    (tmp_path / "d.wav").write_bytes(b"D")
    engine = EchoEngine({b"A": "혈압은 130에 80", b"B": "맥박은 칠십이에요"}, fail=b"D")
    cases = [
        ("a", "normal", "혈압은 130에 80"),
        ("b", "dialect", "맥박은 72예요"),
        ("c", "normal", "음성 파일 없음"),
        ("d", "normal", "실패할 케이스"),
    ]
    report = evaluate(engine, cases, tmp_path)
    assert report["cases"] == 2 and report["missing_audio"] == ["c"] and report["failed"] == ["d"]
    assert report["rows"][0]["cer"] == 0.0 and report["numbers_preserved"] == [3, 3]
    assert set(report["by_category"]) == {"normal", "dialect"}
    assert report["rtf"] == pytest.approx(0.2)  # 0.6초 / 3.0초
    assert "낙관적" in report["note"]
    assert report["hypotheses"] == {"a": "혈압은 130에 80", "b": "맥박은 칠십이에요"}


def test_unavailable_engine_is_not_hidden_as_a_failed_case(tmp_path: Path) -> None:
    (tmp_path / "a.wav").write_bytes(b"A")

    class Broken:
        name = "broken"

        def transcribe(self, audio: bytes, content_type: str) -> Transcription:
            raise SttUnavailable("모델 없음")

    with pytest.raises(SttUnavailable):
        evaluate(Broken(), [("a", "normal", "x")], tmp_path)


def test_find_audio_prefers_known_extensions(tmp_path: Path) -> None:
    assert find_audio(tmp_path, "gs-001") is None
    (tmp_path / "gs-001.mp3").write_bytes(b"x")
    assert find_audio(tmp_path, "gs-001") == tmp_path / "gs-001.mp3"


def test_cli_refuses_without_audio_dir(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["--golden", str(GOLDEN), "--audio-dir", str(tmp_path / "none")])
    assert code == 2 and "tts_mac" in capsys.readouterr().err


def test_cli_end_to_end_with_fake_engine_writes_reports(tmp_path: Path) -> None:
    cases = load_golden(GOLDEN, "dev")[:3]
    texts: dict[bytes, str] = {}
    for case in cases:
        audio = case.id.encode()
        (tmp_path / f"{case.id}.wav").write_bytes(audio)
        texts[audio] = case.transcript
    out, hyp = tmp_path / "r.json", tmp_path / "h.json"
    code = main(
        ["--golden", str(GOLDEN), "--audio-dir", str(tmp_path), "--limit", "3",
         "--out", str(out), "--hyp-out", str(hyp)],
        engine_factory=lambda _: EchoEngine(texts),
    )  # fmt: skip
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["cases"] == 3 and report["cer_micro"] == 0.0
    assert json.loads(hyp.read_text(encoding="utf-8")) == {c.id: c.transcript for c in cases}


# -- TTS (Mac say) -------------------------------------------------------------------


class FakeSay:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[list[str]] = []
        self.fail = fail

    def __call__(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(cmd)
        if cmd[1] == "-v" and cmd[2] == "?":
            return subprocess.CompletedProcess(cmd, 0, "Yuna ko_KR", "")
        if self.fail:
            return subprocess.CompletedProcess(cmd, 1, "", "voice not found")
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"AIFF")
        return subprocess.CompletedProcess(cmd, 0, "", "")


def test_synthesize_writes_files_skips_existing_and_protects_dash_text(tmp_path: Path) -> None:
    say = FakeSay()
    (tmp_path / "gs-002.aiff").write_bytes(b"old")
    written = synthesize(
        tmp_path, [("gs-001", "-v 로 시작하는 문장"), ("gs-002", "있음")], "Yuna", runner=say
    )
    assert [p.name for p in written] == ["gs-001.aiff"]
    assert say.calls[0][-2:] == ["--", "-v 로 시작하는 문장"]  # 옵션으로 읽히지 않는다
    assert (tmp_path / "gs-002.aiff").read_bytes() == b"old"


def test_synthesize_failure_raises_without_partial_silence(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="say 실패"):
        synthesize(tmp_path, [("gs-001", "문장")], "NoSuchVoice", runner=FakeSay(fail=True))


def test_tts_cli_refuses_test_split(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = tts_main(["--split", "test", "--out", str(tmp_path)], runner=FakeSay())
    assert code == 2 and "test 분할" in capsys.readouterr().err


def test_tts_cli_generates_dev_audio_only(tmp_path: Path) -> None:
    say = FakeSay()
    code = tts_main(["--golden", str(GOLDEN), "--out", str(tmp_path), "--limit", "2"], runner=say)
    assert code == 0 and len(list(tmp_path.glob("*.aiff"))) == 2
    dev_ids = {c.id for c in load_golden(GOLDEN, "dev")}
    assert {p.stem for p in tmp_path.glob("*.aiff")} <= dev_ids
