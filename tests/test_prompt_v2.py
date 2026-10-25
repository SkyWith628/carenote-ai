"""추출 프롬프트 v2: v1 불변, 정책 문구, 캐시 분리, --prompt 옵션."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import cast

import pytest

from carenote.config import Settings
from carenote.evaluation.__main__ import main
from carenote.evaluation.loader import load_golden
from carenote.extraction.llm_extractor import LLMExtractor, ParseClient
from carenote.extraction.prompts import (
    DEFAULT_PROMPT_VERSION,
    PROMPT_VERSIONS,
    build_system_prompt,
)

GOLDEN = Path(__file__).parent.parent / "data" / "golden"
# v1 본문은 2026-10-23~26 실측의 기준선이다. 바뀌면 기존 보고서를 재현할 수 없다.
V1_SHA256 = {
    "b0": "3678dc45a8c637dd1fb5b70833bdae15864fd51c24260202e223ed50e40ace81",
    "b1": "0d99a5274fd582eee76d2d4d5969fc601a9cb99a622f4e2ea18ab03871134546",
}


@pytest.mark.parametrize("variant", ["b0", "b1"])
def test_v1_prompt_is_byte_identical_to_the_measured_baseline(variant: str) -> None:
    prompt = build_system_prompt(variant, "v1")  # type: ignore[arg-type]
    assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == V1_SHA256[variant]
    assert build_system_prompt(variant) == prompt  # type: ignore[arg-type]
    assert DEFAULT_PROMPT_VERSION == "v1" and PROMPT_VERSIONS == ("v1", "v2")


V2_POLICY_FRAGMENTS = (
    "추측한 숫자도 넣지 않습니다",  # #1
    "개별 항목을 채우지 않습니다",  # #2
    "마지막 값을 stated",  # #3
    "해소되지 않으면",  # #4
    "마지막 측정값",  # #5
    "앞선 측정은 special_notes",  # #5 (2026-10-26)
    "가족, 화자 본인",  # #6
    "숟가락질 두세 번·거의 못 드심=소량",  # #9
    "최근 7일 이내",  # #7
    "근거가 아닙니다",  # #10
    "필드가 이미 담는 내용은 special_notes에 중복 기록하지 않습니다",  # #11
    "측정을 못 했거나 빠뜨린 사유",  # #11
    "복용 거부 의사",  # #13
    "공=0",  # 숫자 한 글자씩 읽기
)


@pytest.mark.parametrize("variant", ["b0", "b1"])
def test_v2_prompt_carries_the_label_policy(variant: str) -> None:
    prompt = build_system_prompt(variant, "v2")  # type: ignore[arg-type]
    for fragment in V2_POLICY_FRAGMENTS:
        assert fragment in prompt, fragment
    assert "따르지 않습니다" in prompt  # 입력 격리 유지
    assert ("evidence" in prompt) == (variant == "b1")  # 변형 간 누수 없음


def test_v2_drops_the_conflicting_v1_sentences() -> None:
    v1, v2 = build_system_prompt("b0", "v1"), build_system_prompt("b0", "v2")
    assert "과거 낙상도 있음" in v1 and "과거 낙상도 있음" not in v2
    assert "전해 들은 것도 말한 대로 기록" in v1 and "전해 들은 것도" not in v2
    assert v1 != v2


def test_v2_prompt_has_no_dev_or_test_case_sentences() -> None:
    """프롬프트에 골든셋 transcript 문장을 복사하지 않는다 (평가 누수 방지)."""
    prompt = build_system_prompt("b1", "v2")
    for case in load_golden(GOLDEN):
        for sentence in (s.strip() for s in case.transcript.replace("\n", " ").split(".")):
            if len(sentence) >= 12:
                assert sentence not in prompt, (case.id, sentence)
        for field in case.fields:
            for ev in field.evidence:
                if len(ev.text) >= 8:
                    assert ev.text not in prompt, (case.id, ev.text)


class _Fake:
    def __init__(self) -> None:
        self.systems: list[str] = []

    @property
    def messages(self) -> object:
        return self

    def parse(self, **kwargs: object) -> object:
        self.systems.append(str(kwargs["system"]))
        raise AssertionError("캐시 분리 테스트: 호출 자체를 막는다")


def _extractor(version: str, cache: Path, client: object | None = None) -> LLMExtractor:
    return LLMExtractor(
        cast(ParseClient, client or _Fake()),
        variant="b1",
        model="claude-sonnet-5-5",
        effort="medium",
        cache_dir=cache,
        prompt_version=version,  # type: ignore[arg-type]
    )


def test_cache_key_and_system_prompt_follow_the_prompt_version(tmp_path: Path) -> None:
    v1, v2 = _extractor("v1", tmp_path), _extractor("v2", tmp_path)
    assert v1.system_prompt == build_system_prompt("b1", "v1")
    assert v2.system_prompt == build_system_prompt("b1", "v2")
    assert v1._cache_key("가") != v2._cache_key("가")  # pyright: ignore[reportPrivateUsage]
    assert v1.usage_summary()["prompt_version"] == "v1"
    assert v2.usage_summary()["prompt_version"] == "v2"


def test_v1_cache_key_is_unchanged_so_old_cache_still_hits() -> None:
    """기본 버전의 캐시 키 공식이 바뀌지 않았다 (기존 실측 캐시 재사용)."""
    import json

    payload = json.dumps(["claude-sonnet-5-5", "medium", "b1", "v1", "가"], ensure_ascii=False)
    expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    ex = _extractor("v1", Path("."))
    assert ex._cache_key("가") == expected  # pyright: ignore[reportPrivateUsage]


def test_cli_prompt_option_changes_dry_run_cache_hits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """v2는 v1 캐시를 쓰지 않으므로 dry-run에서 캐시 적중이 0이다."""
    base = [
        "--extractor", "llm-b1", "--split", "dev", "--dry-run",
        "--cache-dir", str(tmp_path),
    ]  # fmt: skip
    cache = Path(".cache/extractions")
    assert main([*base, "--prompt", "v1", "--cache-dir", str(cache)], settings=Settings()) == 0
    v1_out = capsys.readouterr().out
    assert "프롬프트 v1" in v1_out
    assert main([*base, "--prompt", "v2", "--cache-dir", str(cache)], settings=Settings()) == 0
    v2_out = capsys.readouterr().out
    assert "프롬프트 v2" in v2_out and "캐시 적중 0건, API 호출 예정 30건" in v2_out


def test_cli_rejects_unknown_prompt_version() -> None:
    with pytest.raises(SystemExit):
        main(["--extractor", "llm-b1", "--prompt", "v3", "--dry-run"])
