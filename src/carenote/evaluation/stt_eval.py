"""골든셋 합성 음성의 STT 정확도 측정 (CER·수치 보존율). dev 분할만 읽는다.

  python -m carenote.evaluation.tts_mac --out data/golden/audio        # Mac에서 음성 생성
  python -m carenote.evaluation.stt_eval --audio-dir data/golden/audio --out OUT.json

받아 적은 글(`hypotheses`)은 보고서에 담아 `--hyp-out`으로도 저장한다. 이 글을 추출기에 넣어
"음성 → STT → 추출"의 최종 필드 정확도를 텍스트 입력과 비교하는 것이 다음 단계다
(docs/stt-design.md 5장). 합성 음성은 낙관적이므로 결과를 실제 정확도로 말하지 않는다.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from carenote.config import Settings, get_settings
from carenote.evaluation.loader import load_golden
from carenote.evaluation.stt_metrics import cer, numbers_preserved
from carenote.stt.engine import SttEngine, SttError, SttUnavailable
from carenote.stt.whisper_engine import FasterWhisperEngine

AUDIO_TYPES = {
    ".aiff": "audio/aiff",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
    ".ogg": "audio/ogg",
}
EXIT_REFUSED = 2
EXIT_ALL_FAILED = 3


def find_audio(audio_dir: Path, case_id: str) -> Path | None:
    """`{id}.{확장자}` 파일을 찾는다."""
    for ext in AUDIO_TYPES:
        path = audio_dir / f"{case_id}{ext}"
        if path.exists():
            return path
    return None


def engine_from_settings(settings: Settings) -> SttEngine:
    """설정으로 faster-whisper 엔진을 만든다 (모델은 첫 호출에서 불러온다)."""
    return FasterWhisperEngine(
        model=settings.stt_model,
        device=settings.stt_device,
        compute_type=settings.stt_compute_type,
        language=settings.stt_language,
        max_seconds=settings.stt_max_seconds,
        download_root=settings.stt_download_root,
        local_files_only=settings.stt_local_files_only,
    )


def evaluate(
    engine: SttEngine, cases: list[tuple[str, str, str]], audio_dir: Path
) -> dict[str, Any]:
    """(id, 유형, 기준 transcript)를 돌려 보고서 dict를 만든다. 음성이 없는 케이스는 건너뛴다."""
    rows: list[dict[str, Any]] = []
    hypotheses: dict[str, str] = {}
    missing: list[str] = []
    failed: list[str] = []
    failures: dict[str, str] = {}
    distance = length = kept = total_numbers = 0
    started = time.perf_counter()
    for case_id, category, reference in cases:
        path = find_audio(audio_dir, case_id)
        if path is None:
            missing.append(case_id)
            continue
        try:
            result = engine.transcribe(path.read_bytes(), AUDIO_TYPES[path.suffix])
        except SttUnavailable:
            raise
        except SttError as exc:
            failed.append(case_id)
            failures[case_id] = str(exc)  # 오디오 내용은 담기지 않는 메시지다
            continue
        c = cer(reference, result.text)
        k, n = numbers_preserved(reference, result.text)
        distance, length, kept, total_numbers = (
            distance + c.distance,
            length + c.ref_length,
            kept + k,
            total_numbers + n,
        )
        hypotheses[case_id] = result.text
        rows.append(
            {
                "id": case_id,
                "category": category,
                "cer": round(c.cer, 4),
                "numbers": [k, n],
                "audio_s": result.duration_s,
                "stt_s": result.elapsed_s,
            }
        )
    by_category: dict[str, list[float]] = {}
    for r in rows:
        by_category.setdefault(r["category"], []).append(r["cer"])
    return {
        "engine": engine.name,
        "cases": len(rows),
        "missing_audio": missing,
        "failed": failed,
        "failures": failures,
        "cer_micro": round(distance / length, 4) if length else None,
        "cer_mean": round(statistics.fmean(r["cer"] for r in rows), 4) if rows else None,
        "numbers_preserved": [kept, total_numbers],
        "by_category": {k: round(statistics.fmean(v), 4) for k, v in sorted(by_category.items())},
        "rtf": round(sum(r["stt_s"] for r in rows) / max(sum(r["audio_s"] for r in rows), 1e-9), 3),
        "wall_s": round(time.perf_counter() - started, 1),
        "rows": rows,
        "hypotheses": hypotheses,
        "note": "합성 음성(깨끗한 음질)은 낙관적 상한이다. 실제 정확도로 말하지 않는다.",
    }


def main(
    argv: list[str] | None = None,
    *,
    settings: Settings | None = None,
    engine_factory: Callable[[Settings], SttEngine] = engine_from_settings,
) -> int:
    """CLI 진입점. settings·engine_factory는 테스트에서 바꿔 끼운다."""
    parser = argparse.ArgumentParser(prog="carenote.evaluation.stt_eval", description=__doc__)
    parser.add_argument("--golden", type=Path, default=Path("data/golden"))
    parser.add_argument("--audio-dir", type=Path, default=Path("data/golden/audio"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None, help="JSON 보고서 경로")
    parser.add_argument("--hyp-out", type=Path, default=None, help="받아 적은 글 JSON 경로")
    args = parser.parse_args(argv)
    if not args.audio_dir.exists():
        print(
            f"실행 거부: 음성 폴더가 없다 ({args.audio_dir}). "
            "`python -m carenote.evaluation.tts_mac`로 먼저 만든다.",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    cases = [(c.id, c.category.value, c.transcript) for c in load_golden(args.golden, "dev")]
    cases = cases[: args.limit]
    try:
        report = evaluate(engine_factory(settings or get_settings()), cases, args.audio_dir)
    except SttError as exc:
        print(f"실행 거부: STT 엔진을 쓸 수 없다 ({exc})", file=sys.stderr)
        return EXIT_REFUSED
    summary = {k: v for k, v in report.items() if k not in ("rows", "hypotheses")}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    exit_code = 0
    if not report["cases"] and report["failed"]:
        reasons = sorted(set(report["failures"].values()))
        print(
            f"실패: 모든 케이스({len(report['failed'])}건)가 실패했다: {reasons}", file=sys.stderr
        )
        exit_code = EXIT_ALL_FAILED
    if args.out is not None:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.hyp_out is not None:
        args.hyp_out.write_text(
            json.dumps(report["hypotheses"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
