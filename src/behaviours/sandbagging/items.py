"""Seeded generator for integer multiplication items.

`make items` calls main() to write data/items.jsonl with every tier.
"""

import json
import random
from pathlib import Path

from behaviours.base import Item

SEED = 20261017
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
    return [
        Item(id=f"t{tier}-{i:03d}", prompt=f"What is {a} × {b}?", target=str(a * b), tier=tier)
        for i, (a, b) in enumerate(sorted(pairs, key=lambda p: rng.random()))
    ]


def write_items(items: list[Item], path: Path = ITEMS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for item in items:
            f.write(json.dumps(item.__dict__) + "\n")


def load_items(tier: int, path: Path = ITEMS_PATH) -> list[Item]:
    with path.open() as f:
        items = [Item(**json.loads(line)) for line in f]
    return [item for item in items if item.tier == tier]


def main() -> None:
    all_items = [item for tier in TIERS for item in generate(tier)]
    write_items(all_items)
    print(f"wrote {len(all_items)} items to {ITEMS_PATH}")
