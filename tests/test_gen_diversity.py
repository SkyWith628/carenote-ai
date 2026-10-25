"""다양성 배정(g4)과 관련 점검 테스트."""

from __future__ import annotations

import json
import re
from collections import Counter

from conftest import GOOD_D1

from carenote.generation.checks import (
    _CLINICAL_FAMILIES,  # pyright: ignore[reportPrivateUsage]
    check_k_weather_openings,
    check_n_diversity,
    run_candidate_checks,
)
from carenote.generation.diversity import OPENING_TYPES, TOPICS, assign_diversity
from carenote.generation.generator import Generator
from carenote.generation.prompts import SYSTEM_PROMPT, build_user_message
from carenote.generation.schema import GeneratedCase
from carenote.generation.spec import Spec


def test_assignment_is_deterministic_and_order_independent(all_specs: list[Spec]) -> None:
    """같은 spec_id 집합이면 입력 순서와 상관없이 같은 배정이 나온다."""
    ids = [s.spec_id for s in all_specs]
    assert assign_diversity(ids) == assign_diversity(reversed(ids))
    assert assign_diversity(ids) == assign_diversity(ids)


def test_assignment_spreads_topics_and_openings_across_all_specs(all_specs: list[Spec]) -> None:
    """40건 안에서 소재는 최대 2번, 소재 쌍은 반복 없이, 첫 문장 유형은 고르게 쓰인다."""
    assignment = assign_diversity(s.spec_id for s in all_specs)
    assert len(assignment) == 40
    topic_use = Counter(t.label for d in assignment.values() for t in d.topics)
    assert max(topic_use.values()) <= 2
    pairs = [tuple(sorted(t.label for t in d.topics)) for d in assignment.values()]
    assert len(set(pairs)) == len(pairs)
    for d in assignment.values():
        assert d.topics[0] != d.topics[1]
    openings = Counter(d.opening_type for d in assignment.values())
    assert set(openings) == set(OPENING_TYPES)
    assert max(openings.values()) - min(openings.values()) <= 1


def test_topics_are_non_clinical_and_numberless() -> None:
    """잡담 소재는 임상 키워드와 숫자를 담지 않는다 (점검 M·E와 충돌 방지)."""
    stems = [stem for family in _CLINICAL_FAMILIES for stem in family]
    for topic in TOPICS:
        text = topic.label + " ".join(topic.keywords)
        assert not any(stem in text for stem in stems), topic.label
        assert not re.search(r"\d", text), topic.label
        assert topic.keywords


def test_prompt_and_message_carry_the_assignment(spec_by_id: dict[str, Spec]) -> None:
    """사용자 메시지에는 첫 문장 유형과 소재 라벨이 들어가고 키워드는 숨겨진다."""
    assignment = assign_diversity(spec_by_id)
    spec = spec_by_id["H3"]
    payload = json.loads(build_user_message(spec, diversity=assignment["H3"]))
    div = payload["diversity"]
    assert div["opening_type"] in OPENING_TYPES and len(div["chatter_topics"]) == 2
    assert "keywords" not in json.dumps(payload, ensure_ascii=False)
    assert "diversity" not in json.loads(build_user_message(spec))
    assert "날씨로 첫 문장을 시작하지 않습니다" in SYSTEM_PROMPT


def test_generator_request_and_cache_key_depend_on_assignment(
    spec_by_id: dict[str, Spec],
) -> None:
    """배정이 다르면 요청 메시지와 캐시 키가 달라진다."""
    spec = spec_by_id["H3"]
    base = assign_diversity(spec_by_id)
    shifted = assign_diversity(["A0", *spec_by_id])  # 맨 앞에 끼워 넣어 정렬 위치를 한 칸씩 민다
    g1 = Generator(None, model="m", effort="medium", diversity=base)  # type: ignore[arg-type]
    g2 = Generator(None, model="m", effort="medium", diversity=shifted)  # type: ignore[arg-type]
    assert base["H3"] != shifted["H3"]
    assert g1.request_message(spec) != g2.request_message(spec)
    assert g1._cache_key(spec, 1, None) != g2._cache_key(spec, 1, None)  # pyright: ignore[reportPrivateUsage]
    assert g1.diversity_for("H3") == base["H3"] and g1.diversity_for("nope") is None


def test_n_warns_when_assigned_topic_is_unused(spec_by_id: dict[str, Spec]) -> None:
    """긴 발화에서 배정된 소재 키워드가 하나도 없으면 경고하고, 있으면 통과한다."""
    assignment = assign_diversity(spec_by_id)
    spec = spec_by_id["D1"]  # 상한 300자
    div = assignment["D1"]
    assert check_n_diversity(spec, GOOD_D1, div).level == "warn"
    keyword = div.topics[0].keywords[0]
    with_topic = GOOD_D1.model_copy(
        update={"transcript": GOOD_D1.transcript + f" {keyword} 얘기도 했어요."}
    )
    assert check_n_diversity(spec, with_topic, div).level == "pass"
    assert check_n_diversity(spec, GOOD_D1, None).level == "pass"


def test_n_is_skipped_for_short_specs(spec_by_id: dict[str, Spec]) -> None:
    """상한 150자 미만 스펙(H4, H5)에는 소재를 요구하지 않는다."""
    assignment = assign_diversity(spec_by_id)
    h4 = spec_by_id["H4"]
    case = GeneratedCase(
        transcript="오늘 방문했어요.", reference_transcript=None, evidence=[], notes=""
    )
    assert check_n_diversity(h4, case, assignment["H4"]).level == "pass"


def test_run_candidate_checks_includes_n_only_with_assignment(
    spec_by_id: dict[str, Spec],
) -> None:
    """배정을 주면 결과 목록에 N이 들어가고, 안 주면 통과로 기록된다."""
    spec = spec_by_id["D1"]
    codes = [r.code for r in run_candidate_checks(spec, GOOD_D1)]
    assert codes[-1] == "N"
    with_div = run_candidate_checks(spec, GOOD_D1, assign_diversity(spec_by_id)["D1"])
    assert with_div[-1].code == "N" and with_div[-1].level == "warn"


def test_k_weather_openings_warn_when_half_or_more_start_with_weather() -> None:
    """첫 문장이 날씨 이야기인 후보가 절반 이상이면 그 후보들에 경고한다."""
    items = [
        ("A", "오늘 날씨가 꽤 쌀쌀했어요. 어르신 댁에 갔고요."),
        ("B", "아침에 비가 와서 우산 들고 갔어요. 별일 없었어요."),
        ("C", "어르신이 문을 열자마자 반겨 주셨어요."),
        ("D", "현관에서 택배 아저씨와 마주쳤어요."),
    ]
    results = check_k_weather_openings(items)
    assert results["A"].level == "warn" and results["B"].level == "warn"
    assert results["C"].level == "pass" and results["D"].level == "pass"
    assert all(r.level == "pass" for r in check_k_weather_openings(items[:2]).values())


def test_m_does_not_flag_apartment_as_pain() -> None:
    """'아파트'는 통증이 아니다 (소재 '아파트 승강기 점검'과 충돌하지 않는다)."""
    from carenote.generation.checks import check_m_new_clinical_facts  # noqa: PLC0415

    assert any(t.label == "아파트 승강기 점검" for t in TOPICS)
    # 임상 키워드가 없는 스펙(D1)에서 '아파트'가 있어도 통과해야 한다
    import conftest  # noqa: PLC0415

    case = conftest.GOOD_D1.model_copy(
        update={"transcript": conftest.GOOD_D1.transcript + " 아파트 승강기 점검이 있었대요."}
    )
    spec = next(s for s in _load_all() if s.spec_id == "D1")
    assert check_m_new_clinical_facts(spec, case).level == "pass"


def _load_all() -> list[Spec]:
    from conftest import SPEC_DIR  # noqa: PLC0415

    from carenote.generation.spec import load_specs  # noqa: PLC0415

    return load_specs(SPEC_DIR)
