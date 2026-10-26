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
from carenote.extraction.schema import LLMExtractionB1

GOLDEN = Path(__file__).parent.parent / "data" / "golden"
# v1 본문은 2026-10-23~26 실측의 기준선이다. 바뀌면 기존 보고서를 재현할 수 없다.
V1_SHA256 = {
    "b0": "3678dc45a8c637dd1fb5b70833bdae15864fd51c24260202e223ed50e40ace81",
    "b1": "0d99a5274fd582eee76d2d4d5969fc601a9cb99a622f4e2ea18ab03871134546",
}


# v2도 측정이 끝난 본문(2026-10-26 dev 실측)이라 고정한다. 바꾸려면 v4를 만든다.
V2_SHA256 = {
    "b0": "99a29a74dd88537b986e7bbc34ff71902b130f9faa3585f49781b774caf8ef4e",
    "b1": "cb2bba04c1620ae3c74eddb4819ec384d6c1333ef45110a5da218b6ca7b1a88e",
}


@pytest.mark.parametrize("variant", ["b0", "b1"])
def test_v2_prompt_is_byte_identical_to_the_measured_text(variant: str) -> None:
    prompt = build_system_prompt(variant, "v2")  # type: ignore[arg-type]
    assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == V2_SHA256[variant]


# v3도 측정이 끝난 본문(2026-10-26 dev 실측과 반복 실행)이라 고정한다. 바꾸려면 v5를 만든다.
V3_SHA256 = {
    "b0": "e0c60b9c848787d7d34cb49f290c85fb9f1adca085d3371aa68da49bff7fa44f",
    "b1": "1c122c5dc2fc95ecdc4420116b660052f5d4e29e364fe32fc9fc5feec1207796",
}


@pytest.mark.parametrize("variant", ["b0", "b1"])
def test_v3_prompt_is_byte_identical_to_the_measured_text(variant: str) -> None:
    prompt = build_system_prompt(variant, "v3")  # type: ignore[arg-type]
    assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == V3_SHA256[variant]


V4_FRAGMENTS = (
    "못 드심·안 드심(정도 표현 없이)",  # 식사 매핑
    "일삼오 → 135",  # 숫자 읽기 새 예시
    "복약 필드가 담는 내용(복용·미복용·중복 복용 의심)의 이유나 정황",  # 중복 금지
    "잊었거나 값을 기억하지 못한다는 말 포함",  # (b)
    "확인·질문을 못 한 항목",  # 확인 누락
    "오늘 복약을 화자가 직접 보거나 직접 들은 경우만 기록합니다",  # 정책 #14
    "추측 표현이 붙어도 화자가 직접 본 기분",  # v3 승계
)


@pytest.mark.parametrize("variant", ["b0", "b1"])
def test_v4_is_v3_plus_exactly_the_five_planned_edits(variant: str) -> None:
    import difflib

    v3 = build_system_prompt(variant, "v3").splitlines()  # type: ignore[arg-type]
    v4 = build_system_prompt(variant, "v4").splitlines()  # type: ignore[arg-type]
    diff = list(difflib.ndiff(v3, v4))
    removed = [d for d in diff if d.startswith("- ")]
    added = [d for d in diff if d.startswith("+ ")]
    assert len(removed) == 4 and len(added) == 6  # 기존 4줄 수정 + (b)는 2줄로 늘고 #14는 1줄 추가
    prompt = "\n".join(v4)
    for fragment in V4_FRAGMENTS:
        assert fragment in prompt, fragment
    assert ("evidence" in prompt) == (variant == "b1")
    assert "따르지 않습니다" in prompt


def test_v4_examples_do_not_reuse_golden_text() -> None:
    """새 예시(일삼오, 칠오)와 가족 예시가 골든셋 transcript에 없다."""
    texts = " ".join(c.transcript for c in load_golden(GOLDEN))
    for example in ("일삼오", "칠오", "가족이 늘 챙긴다"):
        assert example not in texts


@pytest.mark.parametrize("variant", ["b0", "b1"])
def test_v3_adds_only_the_hedged_mood_rule(variant: str) -> None:
    import difflib

    v2 = build_system_prompt(variant, "v2").splitlines()  # type: ignore[arg-type]
    v3 = build_system_prompt(variant, "v3").splitlines()  # type: ignore[arg-type]
    added = [d[2:] for d in difflib.ndiff(v2, v3) if d.startswith("+ ")]
    removed = [d[2:] for d in difflib.ndiff(v2, v3) if d.startswith("- ")]
    assert any("추측 표현이 붙어도 화자가 직접 본 기분" in line for line in added)
    assert any("단 기분(mood)은 아래 mood 규칙을 따릅니다" in line for line in added)
    assert len(removed) == 1 and len(added) == 2  # 한 줄 수정 + mood 규칙 줄 추가
    assert "마지막 값을 stated" in "\n".join(v3)  # 나머지 v2 규칙은 유지


def test_policy_and_blind_guide_allow_hedged_mood() -> None:
    from carenote.generation.candidates import BLIND_GUIDE

    text = (GOLDEN.parent.parent / "docs" / "label-policy.md").read_text(encoding="utf-8")
    assert "추측 표현이 붙어도 화자가 직접 관찰한 기분" in text
    assert "추측 표현이 붙어도 화자가 직접 본 기분" in BLIND_GUIDE


@pytest.mark.parametrize("variant", ["b0", "b1"])
def test_v1_prompt_is_byte_identical_to_the_measured_baseline(variant: str) -> None:
    prompt = build_system_prompt(variant, "v1")  # type: ignore[arg-type]
    assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == V1_SHA256[variant]
    assert PROMPT_VERSIONS == ("v1", "v2", "v3", "v4")


@pytest.mark.parametrize("variant", ["b0", "b1"])
def test_default_prompt_is_v4_and_older_versions_stay_selectable(variant: str) -> None:
    """2026-10-26 v4를 기본 프롬프트 후보로 채택했다. 이전 보고서는 --prompt v1~v3로 재현한다."""
    assert DEFAULT_PROMPT_VERSION == "v4"
    assert build_system_prompt(variant) == build_system_prompt(variant, "v4")  # type: ignore[arg-type]
    assert build_system_prompt(variant) != build_system_prompt(variant, "v1")  # type: ignore[arg-type]


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


@pytest.mark.parametrize("version", ["v2", "v3", "v4"])
def test_prompts_have_no_dev_or_test_case_sentences(version: str) -> None:
    """프롬프트에 골든셋 transcript 문장을 복사하지 않는다 (평가 누수 방지)."""
    prompt = build_system_prompt("b1", version)  # type: ignore[arg-type]
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


def test_cli_prompt_option_is_reported_and_isolates_cache(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--prompt가 dry-run에 표시되고, v1 캐시는 v2에 적중하지 않는다 (임시 캐시, 실제 캐시 무관)."""
    base = ["--extractor", "llm-b1", "--split", "dev", "--dry-run", "--cache-dir", str(tmp_path)]
    case = load_golden(GOLDEN, "dev")[0]
    seeded = _extractor("v1", tmp_path)
    seeded._write_cache(  # pyright: ignore[reportPrivateUsage]
        seeded._cache_key(case.transcript),  # pyright: ignore[reportPrivateUsage]
        LLMExtractionB1(fields=[]),
    )
    assert main([*base, "--prompt", "v1"], settings=Settings()) == 0
    v1_out = capsys.readouterr().out
    assert "프롬프트 v1" in v1_out and "캐시 적중 1건" in v1_out
    assert main([*base, "--prompt", "v2"], settings=Settings()) == 0
    v2_out = capsys.readouterr().out
    assert "프롬프트 v2" in v2_out and "캐시 적중 0건, API 호출 예정 30건" in v2_out


def test_cli_rejects_unknown_prompt_version() -> None:
    with pytest.raises(SystemExit):
        main(["--extractor", "llm-b1", "--prompt", "v9", "--dry-run"])
