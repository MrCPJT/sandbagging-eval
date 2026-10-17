"""Run cells for one behaviour and model, with a printed estimate and budget guard.

Without --approve the runner prints the cell table and cost estimate and exits.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai.log import EvalLog
from inspect_ai.model import Model

from behaviours import get_behaviour
from behaviours.base import Behaviour, Item
from pipeline import budget
from pipeline.task import Cell, build_task

LOG_ROOT = Path("logs")


@dataclass(frozen=True)
class RunPlan:
    n_items: int
    epochs: int
    # "one": first condition, paraphrase 0
    # "tiers": first condition, paraphrase 0, every tier
    # "all": every condition x paraphrase
    cells: str


PLANS: dict[str, RunPlan] = {
    "skeleton": RunPlan(n_items=5, epochs=1, cells="one"),
    "calibrate": RunPlan(n_items=20, epochs=1, cells="tiers"),
    "dry": RunPlan(n_items=20, epochs=1, cells="all"),
    "full": RunPlan(n_items=150, epochs=3, cells="all"),
}

GENERATE_CONFIG = {
    "reasoning_effort": "medium",
    "max_connections": 10,
    "max_retries": 3,
}


def build_cells(behaviour: Behaviour, tier: int, scope: str) -> list[Cell]:
    first = behaviour.conditions[0]
    if scope == "one":
        return [Cell(behaviour.name, first, 0, tier)]
    if scope == "tiers":
        return [Cell(behaviour.name, first, 0, t) for t in behaviour.tiers]
    return [
        Cell(behaviour.name, condition, paraphrase, tier)
        for condition in behaviour.conditions
        for paraphrase in range(behaviour.n_paraphrases)
    ]


def print_plan(
    cells: list[Cell], items: dict[int, list[Item]], epochs: int, model: str, per_call: float
) -> float:
    print(f"model {model}  epochs {epochs}  per-call ${per_call:.5f}")
    print(f"{'cell':<40}{'calls':>8}{'est $':>10}")
    total_calls = 0
    for cell in cells:
        calls = len(items[cell.tier]) * epochs
        total_calls += calls
        print(f"{cell.name:<40}{calls:>8}{calls * per_call:>10.4f}")
    estimate = total_calls * per_call
    print(f"{'total':<40}{total_calls:>8}{estimate:>10.4f}")
    print(f"with {budget.SAFETY_MARGIN}x margin: ${estimate * budget.SAFETY_MARGIN:.4f}")
    return estimate


def run_cells(
    behaviour: Behaviour,
    cells: list[Cell],
    items: dict[int, list[Item]],
    epochs: int,
    model: str | Model,
    log_dir: Path,
    per_call: float = 0.0,
) -> list[EvalLog]:
    """Run cells one at a time, halting before any cell that would break the total cap."""
    logs = []
    model_name = model if isinstance(model, str) else str(model)
    for cell in cells:
        cell_items = items[cell.tier]
        cell_estimate = len(cell_items) * epochs * per_call * budget.SAFETY_MARGIN
        spent = budget.logged_spend(LOG_ROOT)
        if spent + cell_estimate > budget.TOTAL_CAP_USD:
            raise budget.BudgetExceeded(f"halting before {cell.name}: spent ${spent:.2f}")
        task = build_task(behaviour, cell, cell_items, epochs)
        [log] = inspect_eval(task, model=model, log_dir=str(log_dir), **GENERATE_CONFIG)
        cost = budget.log_cost(log)
        if budget.is_priced(model_name):
            budget.record({"cell": cell.name, "model": model_name, "cost_usd": round(cost, 6)})
        print(f"{cell.name}: {log.status}, ${cost:.4f}")
        logs.append(log)
        if log.status != "success":
            break
    return logs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=PLANS)
    parser.add_argument("--behaviour", default="sandbagging")
    parser.add_argument("--model", default="anthropic/claude-haiku-5-5")
    parser.add_argument("--tier", type=int, default=2, help="ignored by calibrate")
    parser.add_argument("--epochs", type=int, help="override the plan's epochs")
    parser.add_argument("--run-cap", type=float, default=budget.RUN_CAP_USD)
    parser.add_argument("--approve", action="store_true", help="start the run")
    args = parser.parse_args(argv)

    plan = PLANS[args.mode]
    epochs = args.epochs or plan.epochs
    behaviour = get_behaviour(args.behaviour)
    cells = build_cells(behaviour, args.tier, plan.cells)
    items = {t: behaviour.items(t)[: plan.n_items] for t in {c.tier for c in cells}}

    per_call, basis = budget.per_call_cost(args.model, LOG_ROOT)
    print(f"per-call cost basis: {basis}")
    estimate = print_plan(cells, items, epochs, args.model, per_call)
    spent = budget.logged_spend(LOG_ROOT)
    print(f"spent so far ${spent:.4f} of ${budget.TOTAL_CAP_USD:.2f}")
    budget.check(estimate, spent, args.run_cap)
    if args.mode == "full" and basis != budget.OWN_LOGS:
        raise budget.BudgetExceeded("a full run needs this model's own token counts; run dry first")

    if not args.approve:
        print("Not started. Re-run with --approve to start.")
        return
    run_cells(behaviour, cells, items, epochs, args.model, LOG_ROOT / args.mode, per_call)
    print(f"spent so far ${budget.logged_spend(LOG_ROOT):.4f}")


if __name__ == "__main__":
    main()
