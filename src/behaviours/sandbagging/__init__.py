from dataclasses import dataclass

from behaviours.base import Item, Outcome
from behaviours.sandbagging import items, prompts, scorer


@dataclass(frozen=True)
class Sandbagging:
    name: str = "sandbagging"
    conditions: tuple[str, ...] = tuple(prompts.CONDITION_SENTENCES)
    n_paraphrases: int = len(prompts.PARAPHRASES)

    def items(self, tier: int) -> list[Item]:
        return items.load_items(tier)

    def system_prompt(self, condition: str, paraphrase: int) -> str:
        return prompts.system_prompt(condition, paraphrase)

    def score(self, completion: str, target: str, stop_reason: str) -> Outcome:
        return scorer.classify(completion, target, stop_reason)


BEHAVIOUR = Sandbagging()
