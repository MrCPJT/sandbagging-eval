"""Seeded generator for integer multiplication items.

`make items` writes data/items.jsonl with every tier at the main seed.
`make items-replication` writes a fresh item set at REPLICATION_SEED.
"""

import json
import random
from pathlib import Path

from behaviours.base import Item

SEED = 20261017
REPLICATION_SEED = 20261018
N_PER_TIER = 150
ITEMS_PATH = Path("data/items.jsonl")

# Operand digit counts per difficulty tier
TIERS: dict[int, tuple[int, int]] = {
    1: (2, 2), 2: (3, 3), 3: (4, 4), 4: (5, 4), 5: (6, 6), 6: (8, 8), 7: (10, 10),
}


def _operand(rng: random.Random, digits: int) -> int:
    return rng.randint(10 ** (digits - 1), 10**digits - 1)


def generate(tier: int, n: int = N_PER_TIER, seed: int = SEED) -> list[Item]:
    rng = random.Random(f"{seed}-{tier}")
    da, db = TIERS[tier]
    pairs: set[tuple[int, int]] = set()
    while len(pairs) < n:
        pairs.add((_operand(rng, da), _operand(rng, db)))
    prefix = "" if seed == SEED else f"s{seed}-"
    return [
        Item(
            id=f"{prefix}t{tier}-{i:03d}",
            prompt=f"What is {a} × {b}?",
            target=str(a * b),
            tier=tier,
        )
        for i, (a, b) in enumerate(sorted(pairs, key=lambda p: rng.random()))
    ]


def items_path(seed: int = SEED) -> Path:
    return ITEMS_PATH if seed == SEED else ITEMS_PATH.with_name(f"items_seed{seed}.jsonl")


def write_items(items: list[Item], path: Path = ITEMS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item in items:
            f.write(json.dumps(item.__dict__) + "\n")


def load_items(tier: int, path: Path = ITEMS_PATH) -> list[Item]:
    with path.open() as f:
        items = [Item(**json.loads(line)) for line in f]
    return [item for item in items if item.tier == tier]


def main(seed: int = SEED, tiers: tuple[int, ...] = tuple(TIERS)) -> None:
    all_items = [item for tier in tiers for item in generate(tier, seed=seed)]
    path = items_path(seed)
    write_items(all_items, path)
    print(f"wrote {len(all_items)} items to {path}")
