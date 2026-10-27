"""골든셋 transcript를 macOS `say`로 합성 음성으로 만든다 (Mac 전용, 외부 호출·비용 없음).

  python -m carenote.evaluation.tts_mac --out data/golden/audio --split dev
  python -m carenote.evaluation.tts_mac --list-voices

합성 음성은 깨끗한 스튜디오 음질이라 실제 방문 녹음(잡음·방언·겹침)보다 STT에 쉽다. 여기서 얻는
정확도는 낙관적 상한이며 "실제 정확도"로 말하면 안 된다. 실제 수급자 음성은 쓰지 않는다.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from carenote.evaluation.loader import load_golden

DEFAULT_VOICE = "Yuna"  # macOS 한국어 음성
Runner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=120)


def synthesize(
    out_dir: Path, items: list[tuple[str, str]], voice: str, *, runner: Runner = _run
) -> list[Path]:
    """(id, 문장) 목록을 `{id}.aiff`로 합성한다. 이미 있는 파일은 건너뛴다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for item_id, text in items:
        target = out_dir / f"{item_id}.aiff"
        if target.exists():
            continue
        # `--`로 문장을 넘겨 "-"로 시작하는 문장이 옵션으로 읽히지 않게 한다
        done = runner(["say", "-v", voice, "-o", str(target), "--", text])
        if done.returncode != 0 or not target.exists():
            raise RuntimeError(f"say 실패: {item_id} ({done.stderr.strip()[:80]})")
        written.append(target)
    return written


def main(argv: list[str] | None = None, *, runner: Runner = _run) -> int:
    """CLI 진입점."""
    parser = argparse.ArgumentParser(prog="carenote.evaluation.tts_mac", description=__doc__)
    parser.add_argument("--golden", type=Path, default=Path("data/golden"))
    parser.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    parser.add_argument("--out", type=Path, default=Path("data/golden/audio"))
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--list-voices", action="store_true")
    args = parser.parse_args(argv)
    if sys.platform != "darwin" and runner is _run:
        print("macOS 전용이다 (`say` 명령이 필요하다).", file=sys.stderr)
        return 2
    if args.list_voices:
        done = runner(["say", "-v", "?"])
        print(done.stdout)
        return done.returncode
    if args.split != "dev":
        print(
            "test 분할 transcript는 음성으로 만들지 않는다: STT 튜닝이 test를 보는 일이 된다. "
            "dev만 쓴다 (docs/stt-design.md).",
            file=sys.stderr,
        )
        return 2
    cases = load_golden(args.golden, "dev")
    items = [(c.id, c.transcript) for c in cases][: args.limit]
    written = synthesize(args.out, items, args.voice, runner=runner)
    print(f"{len(written)}개 생성, {len(items) - len(written)}개는 이미 있음 → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
