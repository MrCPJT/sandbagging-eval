from dataclasses import asdict, dataclass

from inspect_ai import Task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import CORRECT, INCORRECT, Score, Target, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate, system_message

from behaviours.base import Behaviour, Item


@dataclass(frozen=True)
class Cell:
    behaviour: str
    condition: str
    paraphrase: int
    tier: int

    @property
    def name(self) -> str:
        return f"{self.behaviour}-{self.condition}-p{self.paraphrase}-t{self.tier}"


@scorer(metrics=[accuracy(), stderr()])
def behaviour_scorer(behaviour: Behaviour):
    async def score(state: TaskState, target: Target) -> Score:
        outcome = behaviour.score(
            state.output.completion, target.text, state.output.stop_reason
        )
        return Score(
            value=CORRECT if outcome.label == "correct" else INCORRECT,
            answer=outcome.answer,
            metadata={"label": outcome.label, "eval_aware": outcome.eval_aware},
        )

    return score


def build_task(behaviour: Behaviour, cell: Cell, items: list[Item], epochs: int) -> Task:
    """One Inspect task per cell. Cell parameters go in the task metadata."""
    dataset = MemoryDataset(
        [Sample(input=it.prompt, target=it.target, id=it.id) for it in items],
        name=f"{behaviour.name}-t{cell.tier}",
    )
    return Task(
        dataset=dataset,
        solver=[
            system_message(behaviour.system_prompt(cell.condition, cell.paraphrase)),
            generate(),
        ],
        scorer=behaviour_scorer(behaviour),
        epochs=epochs,
        metadata=asdict(cell),
        name=cell.name,
    )
