from dataclasses import dataclass
from typing import Protocol

LABELS = ("correct", "wrong", "refusal", "unparseable")


@dataclass(frozen=True)
class Item:
    id: str
    prompt: str
    target: str
    tier: int


@dataclass(frozen=True)
class Outcome:
    label: str
    answer: str | None
    eval_aware: bool


class Behaviour(Protocol):
    """One behaviour under test: its conditions, prompts, items and scorer."""

    name: str
    conditions: tuple[str, ...]
    n_paraphrases: int

    def items(self, tier: int) -> list[Item]: ...

    def system_prompt(self, condition: str, paraphrase: int) -> str: ...

    def score(self, completion: str, target: str, stop_reason: str) -> Outcome: ...
