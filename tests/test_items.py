import re
from pathlib import Path

import pytest

from behaviours.sandbagging.items import (
    ITEMS_PATH,
    N_PER_TIER,
    TIERS,
    generate,
    load_items,
    write_items,
)

PROMPT = re.compile(r"What is (\d+) × (\d+)\?")


@pytest.mark.parametrize("tier", TIERS)
def test_items_are_correct_and_sized(tier: int) -> None:
    items = generate(tier)
    assert len(items) == N_PER_TIER
    da, db = TIERS[tier]
    for item in items:
        a, b = map(int, PROMPT.fullmatch(item.prompt).groups())
        assert (len(str(a)), len(str(b))) == (da, db)
        assert item.target == str(a * b)
        assert item.tier == tier


@pytest.mark.parametrize("tier", TIERS)
def test_ids_and_prompts_unique(tier: int) -> None:
    items = generate(tier)
    assert len({i.id for i in items}) == len(items)
    assert len({i.prompt for i in items}) == len(items)


def test_generator_is_deterministic() -> None:
    assert generate(2) == generate(2)
    assert generate(2, seed=1) != generate(2)


def test_round_trip_filters_by_tier(tmp_path: Path) -> None:
    path = tmp_path / "items.jsonl"
    write_items(generate(1, n=4) + generate(2, n=3), path)
    assert load_items(1, path) == generate(1, n=4)
    assert load_items(2, path) == generate(2, n=3)


@pytest.mark.skipif(not ITEMS_PATH.exists(), reason="items file not generated")
@pytest.mark.parametrize("tier", TIERS)
def test_committed_items_match_generator(tier: int) -> None:
    assert load_items(tier) == generate(tier)
