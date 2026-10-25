"""다양성 배정(g4)과 관련 점검 테스트."""

from __future__ import annotations

import json
import re
from collections import Counter

from conftest import GOOD_D1

from carenote.evaluation.schema import FieldKey
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
    assert assign_diversity(all_specs) == assign_diversity(reversed(all_specs))
    assert assign_diversity(all_specs) == assign_diversity(all_specs)


def test_assignment_spreads_topics_across_all_specs(all_specs: list[Spec]) -> None:
    """40건 안에서 소재는 최대 2번 쓰이고 소재 쌍은 반복되지 않는다."""
    assignment = assign_diversity(all_specs)
    assert len(assignment) == 40
    topic_use = Counter(t.label for d in assignment.values() for t in d.topics)
    assert max(topic_use.values()) <= 2
    pairs = [tuple(sorted(t.label for t in d.topics)) for d in assignment.values()]
    assert len(set(pairs)) == len(pairs)
    for d in assignment.values():
        assert d.topics[0] != d.topics[1]


def test_opening_types_are_balanced_within_each_split(all_specs: list[Spec]) -> None:
    """첫 문장 유형은 분할 안에서 고르게 쓰이고, 같은 유형은 매번 다른 구체 장면을 받는다."""
    assignment = assign_diversity(all_specs)
    for split in ("dev", "test"):
        members = [assignment[s.spec_id] for s in all_specs if s.split == split]
        types = Counter(d.opening_type for d in members)
        assert set(types) == set(OPENING_TYPES)  # dev 10건도 9개 유형을 모두 쓴다
        assert max(types.values()) - min(types.values()) <= 1
        by_type: dict[str, list[str]] = {}
        for d in members:
            by_type.setdefault(d.opening_type, []).append(d.opening_seed)
        for seeds in by_type.values():
            assert len(set(seeds)) == len(seeds)  # 같은 유형의 장면이 겹치지 않는다


def test_dev_specs_that_collided_in_g5_now_get_different_openings(
    spec_by_id: dict[str, Spec],
) -> None:
    """g5에서 같은 유형을 받아 같은 표현으로 수렴한 H3와 S1은 이제 다른 유형을 받는다."""
    assignment = assign_diversity(spec_by_id.values())
    assert assignment["H3"].opening_type != assignment["S1"].opening_type


def test_every_opening_type_has_distinct_scene_seeds() -> None:
    """유형마다 구체 장면 5개가 있고 서로 다르며 날씨·시각·냉장고를 쓰지 않는다."""
    from carenote.generation.diversity import OPENING_SEEDS, SEEDS_PER_TYPE  # noqa: PLC0415

    assert len(OPENING_SEEDS) == len(OPENING_TYPES)
    all_seeds = [seed for group in OPENING_SEEDS for seed in group]
    assert all(len(group) == SEEDS_PER_TYPE for group in OPENING_SEEDS)
    assert len(set(all_seeds)) == len(all_seeds)
    for seed in all_seeds:
        assert not any(word in seed for word in ("날씨", "비가", "쌀쌀", "냉장고", "시쯤"))


def test_phrasing_hints_describe_the_manner_without_example_sentences() -> None:
    """표현 힌트에는 따옴표 예시·괄호 예시 문장이 없다 (g5 dev: 예시가 3건에 복사됐다)."""
    from carenote.generation.diversity import PHRASING_POOLS  # noqa: PLC0415

    for hints in PHRASING_POOLS.values():
        for hint in hints:
            assert "(" not in hint and "식으로" not in hint, hint
            assert "하시네요" not in hint and "뭐 있냐" not in hint, hint


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
    assignment = assign_diversity(spec_by_id.values())
    spec = spec_by_id["H3"]
    payload = json.loads(build_user_message(spec, diversity=assignment["H3"]))
    div = payload["diversity"]
    assert div["opening_type"] in OPENING_TYPES and len(div["chatter_topics"]) == 2
    assert div["opening_scene"]
    assert "keywords" not in json.dumps(payload, ensure_ascii=False)
    assert "diversity" not in json.loads(build_user_message(spec))
    assert "날씨로 첫 문장을 시작하지 않습니다" in SYSTEM_PROMPT


def test_generator_request_and_cache_key_depend_on_assignment(
    spec_by_id: dict[str, Spec],
) -> None:
    """배정이 다르면 요청 메시지와 캐시 키가 달라진다."""
    spec = spec_by_id["H3"]
    base = assign_diversity(spec_by_id.values())
    extra = spec_by_id["D1"].model_copy(update={"spec_id": "A0"})
    shifted = assign_diversity([extra, *spec_by_id.values()])  # 맨 앞에 끼워 정렬 위치를 민다
    g1 = Generator(None, model="m", effort="medium", diversity=base)  # type: ignore[arg-type]
    g2 = Generator(None, model="m", effort="medium", diversity=shifted)  # type: ignore[arg-type]
    assert base["H3"] != shifted["H3"]
    assert g1.request_message(spec) != g2.request_message(spec)
    assert g1._cache_key(spec, 1, None) != g2._cache_key(spec, 1, None)  # pyright: ignore[reportPrivateUsage]
    assert g1.diversity_for("H3") == base["H3"] and g1.diversity_for("nope") is None


def test_n_warns_when_assigned_topic_is_unused(spec_by_id: dict[str, Spec]) -> None:
    """긴 발화에서 배정된 소재 키워드가 하나도 없으면 경고하고, 있으면 통과한다."""
    assignment = assign_diversity(spec_by_id.values())
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
    assignment = assign_diversity(spec_by_id.values())
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
    without = {r.code: r for r in run_candidate_checks(spec, GOOD_D1)}
    assert without["N"].level == "pass" and without["O"].level == "pass"
    with_div = {
        r.code: r
        for r in run_candidate_checks(spec, GOOD_D1, assign_diversity(spec_by_id.values())["D1"])
    }
    assert with_div["N"].level == "warn"


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


def test_field_order_is_a_non_canonical_permutation_and_deterministic(
    all_specs: list[Spec],
) -> None:
    """항목 언급 순서는 6개 항목의 순열이고 스키마 순서와 다르며 같은 입력이면 같다."""
    from carenote.generation.diversity import CANONICAL_ORDER  # noqa: PLC0415

    first = assign_diversity(all_specs)
    second = assign_diversity(reversed(all_specs))
    for sid, div in first.items():
        assert set(div.field_order) == set(CANONICAL_ORDER) and len(div.field_order) == 6
        assert div.field_order != CANONICAL_ORDER
        assert div.field_order == second[sid].field_order
    assert len({d.field_order for d in first.values()}) > 20  # 40건이 서로 다른 순서를 폭넓게 쓴다


def test_phrasing_hints_only_for_plain_stated_normal_values(
    spec_by_id: dict[str, Spec],
) -> None:
    """정상 값(없음·복용·전량/절반/소량)을 말하는 항목에만 표현 방식이 배정된다."""
    assignment = assign_diversity(spec_by_id.values())
    d1 = dict(assignment["D1"].phrasing)  # 식사 전량, 복용, 낙상 없음 모두 단순 보고
    assert set(d1) == {FieldKey.MEAL_INTAKE, FieldKey.MEDICATION_TAKEN, FieldKey.FALL_EVENT}
    r2 = dict(assignment["R2"].phrasing)  # 식사 미섭취는 해당 풀이 없다
    assert FieldKey.MEAL_INTAKE not in r2
    h4 = dict(assignment["H4"].phrasing)  # 거의 비어 있는 발화: 정답이 전부 missing
    assert h4 == {}
    payload = assignment["D1"].to_payload()
    assert payload["field_order"] and set(payload["phrasing_hints"]) == {  # type: ignore[arg-type]
        "meal_intake",
        "medication_taken",
        "fall_event",
    }


def test_specs_no_longer_ask_for_lunch_on_morning_visits(all_specs: list[Spec]) -> None:
    """오전(12시 이전) 방문 스펙의 scene·must_include에 '점심'이 남아 있지 않다 (20건 일반화)."""
    for spec in all_specs:
        visit = spec.gold[FieldKey.VISIT_START].value
        if visit and int(visit.split(":")[0]) < 12:
            assert "점심" not in spec.scene + " ".join(spec.must_include), spec.spec_id


def test_elder_quote_hints_require_the_fact_to_be_explicit() -> None:
    """어르신 발언 전달 힌트는 정답 사실이 그 문장에서 분명히 읽히게 요구한다 (g6 H3·C2)."""
    from carenote.evaluation.schema import FieldKey  # noqa: PLC0415
    from carenote.generation.diversity import PHRASING_POOLS  # noqa: PLC0415

    fall = [h for h in PHRASING_POOLS[(FieldKey.FALL_EVENT, "없음")] if "어르신이 한 말" in h]
    med = [h for h in PHRASING_POOLS[(FieldKey.MEDICATION_TAKEN, "복용")] if "어르신이 한 말" in h]
    assert fall and all("분명히 읽히게" in h for h in fall)
    assert med and all("분명히 읽히게" in h for h in med)


def test_phrasing_hints_are_distinct_within_a_split_when_the_pool_allows(
    all_specs: list[Spec],
) -> None:
    """같은 분할·항목·값의 스펙은 풀 크기 이하라면 서로 다른 표현 방식을 받는다 (g7 수렴 대응)."""
    from collections import defaultdict  # noqa: PLC0415

    from carenote.generation.diversity import PHRASING_POOLS  # noqa: PLC0415

    assignment = assign_diversity(all_specs)
    slots: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    for spec in all_specs:
        for key, hint in assignment[spec.spec_id].phrasing:
            slots[(spec.split, key.value, spec.gold[key].value or "")].append(hint)
    assert slots
    for (_, key, value), hints in slots.items():
        pool = PHRASING_POOLS[(FieldKey(key), value)]
        window = hints[: len(pool)]
        assert len(set(window)) == len(window), (key, value)


def test_dev_specs_that_shared_the_elder_quote_hint_in_g7_now_differ(
    spec_by_id: dict[str, Spec],
) -> None:
    """H3·C2·S1이 모두 '어르신이 한 말' 낙상 힌트를 받던 g7과 달리 이제 한 건만 받는다."""
    assignment = assign_diversity(spec_by_id.values())
    elder = [
        sid
        for sid in ("H3", "C2", "S1")
        if "어르신이 한 말" in dict(assignment[sid].phrasing).get(FieldKey.FALL_EVENT, "")
    ]
    assert len(elder) <= 1
