"""Sentinel LLM 보조 단계 테스트. 가짜 클라이언트만 쓰며 API를 호출하지 않는다.

test 분할은 읽지 않는다. 겹침·회귀 검사는 dev 자료(골든 dev, R-set dev)만 쓴다.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import anthropic
import httpx2
import pytest
from pydantic import BaseModel

import carenote.evaluation.sentinel_llm_run as run
from carenote.config import Settings
from carenote.evaluation.extractor import PredictedField
from carenote.evaluation.loader import SplitChoice, load_golden
from carenote.evaluation.schema import FieldKey, Flag, GoldenCase
from carenote.extraction.flagger import FlaggingExtractor
from carenote.extraction.llm_extractor import ParseClient, ParseResponse
from carenote.extraction.pricing import BudgetExceeded, CostGuard, price_for
from carenote.extraction.sentinel import Signal, load_rules, parse_rules, scan_text
from carenote.extraction.sentinel_llm import (
    CATEGORIES,
    MAX_SIGNALS,
    SENTINEL_LLM_PROMPT_VERSION,
    SYSTEM_PROMPT,
    LLMSentinelOutput,
    LLMSignal,
    RawSignal,
    SentinelAssist,
    build_user_message,
    merge_signals,
    verify_signals,
)

ROOT = Path(__file__).parent.parent
GOLDEN = ROOT / "data" / "golden"
RSET = GOLDEN / "rset"
RULES = load_rules()

TEXT = (
    "방문했어요. 어르신이 계단 아래에서 굴러떨어지셨고 그 뒤로 어깨를 못 올리세요. 점심은 드셨어요."
)
QUOTE = "굴러떨어지셨고 그 뒤로 어깨를 못 올리세요"


def raw(
    quote: str = QUOTE,
    category: str = "낙상 후",
    subject: str = "client",
    timing: str = "current",
    negated: bool = False,
) -> RawSignal:
    return RawSignal(category, quote, subject, timing, negated)


def checks(text: str, *items: RawSignal) -> list[str]:
    return [r.check for r in verify_signals(text, items, RULES).rejected]


# ---- verify_signals: V1~V6 ----------------------------------------------------------------


def test_valid_signal_is_accepted() -> None:
    result = verify_signals(TEXT, [raw()], RULES)
    assert result.rejected == []
    (sig,) = result.accepted
    assert sig.quote == QUOTE and sig.category == "낙상 후"


def test_accepted_signal_fields() -> None:
    (sig,) = verify_signals(TEXT, [raw()], RULES).accepted
    assert sig.source == "llm" and sig.provisional is True
    assert sig.severity is Flag.RISK and sig.needs_expert is False
    assert sig.rule_id == "llm:낙상 후"


def test_circumstance_signal_needs_expert() -> None:
    text = "팔뚝에 손가락 모양 멍이 있었어요."
    (sig,) = verify_signals(text, [raw("팔뚝에 손가락 모양 멍이 있었어요", "정황")], RULES).accepted
    assert sig.needs_expert is True


def test_v1_quote_not_in_transcript() -> None:
    assert checks(TEXT, raw("계단에서 넘어지셔서 어깨가 아프다고")) == ["V1"]


def test_v1_empty_quote() -> None:
    assert checks(TEXT, raw("")) == ["V1"]
    assert checks(TEXT, raw("   ")) == ["V1"]


def test_v1_whitespace_differences_are_normalized() -> None:
    (sig,) = verify_signals(
        TEXT, [raw("굴러떨어지셨고  그 뒤로\n어깨를 못 올리세요")], RULES
    ).accepted
    assert sig.quote == QUOTE


def test_v1_unicode_normalization() -> None:
    decomposed = unicodedata.normalize("NFD", QUOTE)
    assert decomposed != QUOTE
    assert len(verify_signals(TEXT, [raw(decomposed)], RULES).accepted) == 1


def test_v1_paraphrase_is_rejected() -> None:
    assert checks(TEXT, raw("굴러떨어진 뒤 어깨를 못 올림")) == ["V1"]


def test_v2_too_short() -> None:
    assert checks(TEXT, raw("어깨를")) == ["V2"]


def test_v2_too_long() -> None:
    long_text = "가" * 130 + "."
    assert checks(long_text, raw("가" * 130)) == ["V2"]


def test_v2_too_many_sentence_breaks() -> None:
    text = "첫째 문장이에요. 둘째 문장이에요. 셋째 문장이에요. 넷째 문장이에요."
    assert checks(text, raw(text)) == ["V2"]


def test_v2_trailing_period_is_not_a_break() -> None:
    text = "어르신 발이 퉁퉁 부어 있었어요."
    assert (
        len(
            verify_signals(
                text, [raw("어르신 발이 퉁퉁 부어 있었어요.", "신체 변화")], RULES
            ).accepted
        )
        == 1
    )


def test_v3_unknown_category() -> None:
    assert checks(TEXT, raw(category="기타")) == ["V3"]


def test_v4_other_person_is_rejected() -> None:
    assert checks(TEXT, raw(subject="other")) == ["V4"]


def test_v4_unclear_subject_is_accepted() -> None:
    assert len(verify_signals(TEXT, [raw(subject="unclear")], RULES).accepted) == 1


@pytest.mark.parametrize("timing", ["past", "hypothetical"])
def test_v4_non_current_timing_is_rejected(timing: str) -> None:
    assert checks(TEXT, raw(timing=timing)) == ["V4"]


def test_v4_negated_is_rejected() -> None:
    assert checks(TEXT, raw(negated=True)) == ["V4"]


def test_v5_third_party_in_head_context() -> None:
    text = "사위가 숨이 차서 병원에 갔다고 하셨어요."
    assert checks(text, raw("숨이 차서 병원에 갔다고", "흉부·호흡")) == ["V5"]


def test_v5_negation_after_quote() -> None:
    text = "어르신은 가슴이 답답하지는 않으셨어요."
    assert checks(text, raw("가슴이 답답", "흉부·호흡")) == ["V5"]


def test_v5_conditional_advice_after_quote() -> None:
    text = "혹시 팔에 힘이 빠지거나 말이 어눌해지면 바로 119에 전화하시라고 했어요."
    assert checks(text, raw("팔에 힘이 빠지", "신경·의식")) == ["V5"]


def test_v5_past_event_in_head_context() -> None:
    text = "작년에 사흘째 식사를 거부하신 적이 있다고 해요."
    assert checks(text, raw("사흘째 식사를 거부", "식사·수분")) == ["V5"]


def test_v5_guards_do_not_see_inside_the_quote() -> None:
    """인용 안쪽의 부정은 V4의 negated에 맡긴다(V5는 인용 바깥 문맥만 본다)."""
    text = "어르신은 가슴이 답답하지는 않으셨어요."
    assert (
        len(
            verify_signals(text, [raw("가슴이 답답하지는 않으셨어요", "흉부·호흡")], RULES).accepted
        )
        == 1
    )


def test_v5_applies_the_categorys_own_guards() -> None:
    """식사·수분 규칙에는 past_event 가드가 있고 신경·의식 규칙에는 없다."""
    past = "작년에 말씀이 어눌해진 적이 있어요."
    assert (
        len(verify_signals(past, [raw("말씀이 어눌해진 적이 있어요", "신경·의식")], RULES).accepted)
        == 1
    )


def test_v6_duplicate_quote() -> None:
    assert checks(TEXT, raw(), raw()) == ["V6"]


def test_v6_cap() -> None:
    text = " ".join(f"{n}번째로 발이 퉁퉁 부어 있었어요." for n in range(1, 9))
    items = [raw(f"{n}번째로 발이 퉁퉁 부어 있었어요", "신체 변화") for n in range(1, 9)]
    result = verify_signals(text, items, RULES)
    assert len(result.accepted) == MAX_SIGNALS
    assert [r.check for r in result.rejected] == ["V6"] * (8 - MAX_SIGNALS)


def test_mixed_batch_keeps_only_valid_with_reasons() -> None:
    result = verify_signals(
        TEXT, [raw(), raw("지어낸 문장이에요 정말로"), raw(subject="other")], RULES
    )
    assert [s.quote for s in result.accepted] == [QUOTE]
    assert [r.check for r in result.rejected] == ["V1", "V4"]
    assert all(r.reason for r in result.rejected)


def test_accepted_quotes_are_substrings_of_normalized_transcript() -> None:
    padded = "방문했어요.   어르신이 계단 아래에서\n굴러떨어지셨고 그 뒤로 어깨를 못 올리세요.  끝"
    result = verify_signals(padded, [raw("굴러떨어지셨고   그 뒤로 어깨를 못 올리세요")], RULES)
    assert all(s.quote in " ".join(padded.split()) for s in result.accepted)


# ---- merge_signals ------------------------------------------------------------------------


def rule_signal(category: str, quote: str, matched: str) -> Signal:
    return Signal("r", category, Flag.RISK, quote, matched, True, False)


def llm_signal(category: str, quote: str) -> Signal:
    return Signal("llm:x", category, Flag.RISK, quote, quote, True, False, source="llm")


def test_merge_overlapping_becomes_rule_plus_llm() -> None:
    rule = rule_signal("신체 변화", "발이 퉁퉁 부어 있었어요.", "발이 퉁퉁 부어")
    merged = merge_signals([rule], [llm_signal("신체 변화", "발이 퉁퉁 부어 있었어요")])
    assert len(merged) == 1 and merged[0].source == "rule+llm"


def test_merge_keeps_llm_only_signal() -> None:
    rule = rule_signal("신체 변화", "발이 퉁퉁 부어 있었어요.", "발이 퉁퉁 부어")
    merged = merge_signals([rule], [llm_signal("신체 변화", "엉덩이 상처에서 진물이 나요")])
    assert [m.source for m in merged] == ["rule", "llm"]


def test_merge_different_category_is_not_merged() -> None:
    rule = rule_signal("신체 변화", "발이 퉁퉁 부어 있었어요.", "발이 퉁퉁 부어")
    merged = merge_signals([rule], [llm_signal("낙상 후", "발이 퉁퉁 부어 있었어요")])
    assert len(merged) == 2


# ---- SentinelAssist (가짜 클라이언트) ------------------------------------------------------------


@dataclass(frozen=True)
class _Usage:
    input_tokens: int = 2000
    output_tokens: int = 100
    cache_read_input_tokens: int | None = None


@dataclass(frozen=True)
class _Response:
    parsed_output: object | None
    stop_reason: str | None = "end_turn"
    usage: _Usage = field(default_factory=_Usage)


class _FakeMessages:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, object]] = []

    def parse(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str,
        messages: list[dict[str, str]],
        output_format: type[BaseModel],
        output_config: dict[str, str],
    ) -> ParseResponse:
        self.calls.append(
            {
                "model": model,
                "system": system,
                "messages": messages,
                "output_format": output_format,
                "output_config": output_config,
            }
        )
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return cast(ParseResponse, self.outcome)


class _FakeClient:
    def __init__(self, outcome: object) -> None:
        self.messages = _FakeMessages(outcome)


def signal(**kw: object) -> LLMSignal:
    base: dict[str, object] = {
        "category": "낙상 후",
        "quote": QUOTE,
        "subject": "client",
        "timing": "current",
        "negated": False,
    }
    base.update(kw)
    return LLMSignal.model_validate(base)


def ok_response(*signals: LLMSignal, **kw: object) -> _Response:
    return _Response(LLMSentinelOutput(signals=list(signals)), **kw)  # type: ignore[arg-type]


def make(
    outcome: object, tmp: Path | None = None, **kw: object
) -> tuple[SentinelAssist, _FakeMessages]:
    client = _FakeClient(outcome)
    assist = SentinelAssist(
        cast(ParseClient, client),
        model="claude-haiku-5-5",
        effort="low",
        rules=RULES,
        cache_dir=tmp,
        **kw,  # type: ignore[arg-type]
    )
    return assist, client.messages


def test_assess_sends_expected_request_and_returns_signals() -> None:
    assist, fake = make(ok_response(signal()))
    result = assist.assess(TEXT)
    assert result.status == "ok" and [s.quote for s in result.signals] == [QUOTE]
    (call,) = fake.calls
    assert call["model"] == "claude-haiku-5-5" and call["system"] == SYSTEM_PROMPT
    assert call["output_format"] is LLMSentinelOutput
    assert call["output_config"] == {"effort": "low"}
    assert call["messages"] == [{"role": "user", "content": build_user_message(TEXT)}]


def test_user_message_wraps_transcript_as_data() -> None:
    message = build_user_message("안녕")
    assert message.startswith("<transcript>") and message.endswith("</transcript>")


def test_empty_signal_list_is_ok_not_unavailable() -> None:
    result = make(ok_response())[0].assess(TEXT)
    assert result.status == "ok" and result.signals == []


def test_cache_hit_skips_second_call(tmp_path: Path) -> None:
    assist, fake = make(ok_response(signal()), tmp_path)
    assist.assess(TEXT)
    again = assist.assess(TEXT)
    assert len(fake.calls) == 1 and again.status == "cache" and len(again.signals) == 1
    assert assist.stats.cache_hits == 1 and assist.is_cached(TEXT)


def test_cache_stores_raw_output_so_verification_can_change(tmp_path: Path) -> None:
    """검증 전 원본을 저장하므로, 같은 캐시에 다른 규칙을 적용해도 재호출하지 않는다."""
    bad = signal(quote="지어낸 인용이에요 정말로")
    assist, fake = make(ok_response(bad), tmp_path)
    assert assist.assess(TEXT).signals == []
    assert len(assist.assess(TEXT).rejected) == 1 and len(fake.calls) == 1


def test_api_error_is_unavailable_not_empty(tmp_path: Path) -> None:
    err = anthropic.APIConnectionError(request=httpx2.Request("POST", "https://example.invalid"))
    assist, _ = make(err, tmp_path)
    result = assist.assess(TEXT)
    assert result.status == "unavailable" and "API 오류" in result.reason
    assert assist.stats.failures == 1 and not assist.is_cached(TEXT)


@pytest.mark.parametrize("stop", ["refusal", "max_tokens"])
def test_unusable_stop_reason_is_unavailable(stop: str) -> None:
    result = make(ok_response(signal(), stop_reason=stop))[0].assess(TEXT)
    assert result.status == "unavailable" and stop in result.reason


def test_unparsed_output_is_unavailable() -> None:
    assert make(_Response(None))[0].assess(TEXT).status == "unavailable"


def test_budget_guard_stops_run() -> None:
    price = price_for("claude-haiku-5-5")
    assert price is not None
    guard = CostGuard(0.0001, price)
    assist, _ = make(ok_response(), guard=guard)
    assist.assess(TEXT)
    with pytest.raises(BudgetExceeded):
        assist.assess(TEXT + " 다른 입력")


def test_run_tag_changes_cache_key(tmp_path: Path) -> None:
    a, _ = make(ok_response(), tmp_path)
    b, _ = make(ok_response(), tmp_path, run_tag="r2")
    a.assess(TEXT)
    assert a.is_cached(TEXT) and not b.is_cached(TEXT)


def test_usage_summary_reports_cost() -> None:
    assist, _ = make(ok_response())
    assist.assess(TEXT)
    summary = assist.usage_summary()
    assert summary["calls"] == 1 and summary["prompt_version"] == SENTINEL_LLM_PROMPT_VERSION
    assert isinstance(summary["estimated_cost_usd"], float) and summary["estimated_cost_usd"] > 0


# ---- Consensus (FlaggingExtractor) --------------------------------------------------------------


class _NoFields:
    def extract(self, transcript: str) -> list[PredictedField]:
        return []


def notes_flag(extractor: FlaggingExtractor) -> Flag:
    return next(f for f in extractor.extract(TEXT) if f.key is FieldKey.SPECIAL_NOTES).flag


def test_llm_only_signal_raises_special_notes_to_risk() -> None:
    assert scan_text(TEXT, RULES) == []  # 규칙은 이 문장을 못 본다
    assist, _ = make(ok_response(signal()))
    extractor = FlaggingExtractor(_NoFields(), RULES, assist=assist)
    assert notes_flag(extractor) is Flag.RISK
    sources = [s.source for s in extractor.signals_for(TEXT)]
    assert sources == ["llm"]


def test_unavailable_assist_leaves_rule_flags_unchanged() -> None:
    assist, _ = make(_Response(None))
    extractor = FlaggingExtractor(_NoFields(), RULES, assist=assist)
    assert notes_flag(extractor) is Flag.OK
    result = extractor.assist_result_for(TEXT)
    assert result is not None and result.status == "unavailable"


def test_assist_without_rules_is_refused() -> None:
    assist, _ = make(ok_response())
    with pytest.raises(ValueError, match="rules"):
        FlaggingExtractor(_NoFields(), None, assist=assist)


def test_no_assist_means_no_assist_result() -> None:
    extractor = FlaggingExtractor(_NoFields(), RULES)
    extractor.extract(TEXT)
    assert extractor.assist_result_for(TEXT) is None


def test_rule_and_llm_overlap_is_marked_rule_plus_llm() -> None:
    text = "어르신 두 다리가 퉁퉁 부어서 바지가 안 올라가요."
    assist, _ = make(ok_response(signal(category="신체 변화", quote="두 다리가 퉁퉁 부어서")))
    extractor = FlaggingExtractor(_NoFields(), RULES, assist=assist)
    extractor.extract(text)
    assert [s.source for s in extractor.signals_for(text)] == ["rule+llm"]


# ---- 프롬프트 -----------------------------------------------------------------------------------


def test_prompt_is_pinned_by_hash() -> None:
    """프롬프트를 바꾸면 이 해시와 SENTINEL_LLM_PROMPT_VERSION을 함께 올린다."""
    digest = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    assert (SENTINEL_LLM_PROMPT_VERSION, digest) == (
        "s1",
        "3bd62a938dc54a21c5e44e86cc53ddb8b51800ede833d1344f641e27d115bd53",
    )


def test_categories_match_rule_file() -> None:
    assert set(CATEGORIES) == {r.category for r in RULES.text_rules}


def test_prompt_names_every_category() -> None:
    assert all(f"- {c}:" in SYSTEM_PROMPT for c in CATEGORIES)


def test_prompt_has_no_overlap_with_dev_transcripts() -> None:
    """프롬프트 예시가 평가 자료의 문장을 베끼지 않았는지 본다(dev 자료만)."""

    def shingles(text: str, n: int = 10) -> set[str]:
        flat = "".join(text.split())
        return {flat[i : i + n] for i in range(len(flat) - n + 1)}

    corpus = [c.transcript for c in load_golden(GOLDEN, "dev")]
    for spec_id in json.loads((RSET / "specs" / "probe_K.json").read_text(encoding="utf-8"))[
        "specs"
    ]:
        if spec_id["split"] == "dev":
            path = RSET / "candidates" / f"{spec_id['spec_id']}.json"
            corpus.append(str(json.loads(path.read_text(encoding="utf-8"))["transcript"]))
    prompt = shingles(SYSTEM_PROMPT)
    leaked = [t[:30] for t in corpus if prompt & shingles(t)]
    assert leaked == []


# ---- CLI (가짜 클라이언트, dev만) ----


class _CountingClient:
    """항상 빈 신호를 돌려주고 호출 수를 센다."""

    def __init__(self, usage: _Usage | None = None) -> None:
        self.calls = 0
        self._usage = usage or _Usage()
        self.messages = self

    def parse(self, **_: object) -> ParseResponse:
        self.calls += 1
        return cast(ParseResponse, _Response(LLMSentinelOutput(signals=[]), usage=self._usage))


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    return Settings()


def cli_args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--sentinel-llm",
        "haiku",
        "--golden",
        str(GOLDEN),
        "--rset-dir",
        str(RSET),
        "--cache-dir",
        str(tmp_path / "cache"),
        *extra,
    ]


def no_client(_: Settings) -> ParseClient:
    raise AssertionError("클라이언트를 만들면 안 되는 경로다")


def test_cli_dry_run_makes_no_calls(settings: Settings, tmp_path: Path) -> None:
    code = run.main(cli_args(tmp_path, "--dry-run"), settings=settings, client_factory=no_client)
    assert code == 0 and not (tmp_path / "cache").exists()


def test_cli_requires_approval(settings: Settings, tmp_path: Path) -> None:
    assert (
        run.main(cli_args(tmp_path), settings=settings, client_factory=no_client)
        == run.EXIT_REFUSED
    )


def test_cli_refuses_approval_below_estimate(settings: Settings, tmp_path: Path) -> None:
    args = cli_args(tmp_path, "--approve-cost", "0.001")
    assert run.main(args, settings=settings, client_factory=no_client) == run.EXIT_REFUSED


def test_cli_has_no_way_to_ask_for_test_split(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        run.main(cli_args(tmp_path, "--split", "test", "--dry-run"))


def test_cli_full_run_with_fake_client(settings: Settings, tmp_path: Path) -> None:
    client = _CountingClient()
    out = tmp_path / "report.json"
    args = cli_args(tmp_path, "--approve-cost", "0.06", "--out", str(out))
    code = run.main(args, settings=settings, client_factory=lambda _: cast(ParseClient, client))
    assert code == 0 and client.calls == 50
    report = json.loads(out.read_text(encoding="utf-8"))
    assert len(report["rows"]) == 50
    assert {r["origin"] for r in report["rows"]} == {"golden", "rset"}
    again = run.main(args, settings=settings, client_factory=no_client)  # 전부 캐시
    assert again == 0


def test_cli_stops_when_budget_is_hit(settings: Settings, tmp_path: Path) -> None:
    client = _CountingClient(_Usage(input_tokens=5_000_000, output_tokens=0))
    args = cli_args(tmp_path, "--approve-cost", "0.06")
    code = run.main(args, settings=settings, client_factory=lambda _: cast(ParseClient, client))
    assert code == run.EXIT_BUDGET and client.calls == 1


# ---- 항목 로딩: dev만 --------------------------------------------------------------------------


def test_load_items_reads_only_the_dev_split(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []
    real = run.load_golden

    def spy(path: Path, split: SplitChoice = "all") -> list[GoldenCase]:
        seen.append(split)
        return real(path, split)

    monkeypatch.setattr(run, "load_golden", spy)
    run.load_items(GOLDEN, RSET, "both")
    assert seen == ["dev"]


def test_load_items_groups_and_counts() -> None:
    items = run.load_items(GOLDEN, RSET, "both")
    counts = {g: sum(1 for i in items if i.group == g) for g in {i.group for i in items}}
    assert counts == {
        "lexicon": 5,
        "novel": 4,
        "circ": 1,
        "neg": 10,
        "golden-pos": 3,
        "golden-neg": 27,
    }
    assert [i.item_id for i in items if i.gray] == ["K25", "K26"]


def test_load_items_rset_ids_are_all_dev_specs() -> None:
    specs = json.loads((RSET / "specs" / "probe_K.json").read_text(encoding="utf-8"))["specs"]
    dev_ids = {s["spec_id"] for s in specs if s["split"] == "dev"}
    items = run.load_items(GOLDEN, RSET, "rset-dev")
    assert {i.item_id for i in items} == dev_ids and len(dev_ids) == 20


def test_load_items_missing_candidate_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "specs").mkdir()
    shutil.copy(RSET / "specs" / "probe_K.json", tmp_path / "specs")
    with pytest.raises(ValueError, match="후보가 없다"):
        run.load_items(GOLDEN, tmp_path, "rset-dev")


def test_wilson_interval() -> None:
    assert run.wilson(0, 0) == (0, 0)
    assert run.wilson(9, 9) == (70, 100)
    assert run.wilson(0, 10) == (0, 28)


def test_summarize_labels_groups() -> None:
    rows = [
        {"group": "novel", "rules": False, "llm": True, "union": True, "gray": False},
        {"group": "neg", "rules": False, "llm": False, "union": False, "gray": False},
    ]
    text = "\n".join(run.summarize(rows))
    assert "양성 어휘 밖(R-set) | 0/1" in text and "음성 오발동(R-set)" in text


def test_rules_object_is_reusable_between_modules() -> None:
    assert parse_rules("text_rules: []\n").text_rules == ()
