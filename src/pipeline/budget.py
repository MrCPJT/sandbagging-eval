"""Cost estimates, spend tracking from Inspect logs, and hard caps."""

import json
from datetime import datetime, timezone
from pathlib import Path

from inspect_ai.log import EvalLog, list_eval_logs, read_eval_log

# USD per million tokens (input, output). Anthropic first-party rates, October 2026.
# Haiku 5.5 rates apply to prompts up to 100K tokens.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "anthropic/claude-haiku-5-5": (0.10, 0.50),
    "anthropic/claude-sonnet-5-5": (2.00, 10.00),
}

TOTAL_CAP_USD = 100.0
RUN_CAP_USD = 40.0

# Per-call token prior (input, output incl. thinking) used before any logs exist
PRIOR_TOKENS = (300, 3000)

# Estimates are multiplied by this before any cap check
SAFETY_MARGIN = 1.25

SPEND_PATH = Path("results/spend.json")


class BudgetExceeded(RuntimeError):
    pass


def is_priced(model: str) -> bool:
    return model in PRICES_PER_MTOK


def token_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost in USD. Unpriced models (e.g. mockllm) cost nothing."""
    if not is_priced(model):
        return 0.0
    p_in, p_out = PRICES_PER_MTOK[model]
    return (input_tokens * p_in + output_tokens * p_out) / 1e6


def log_cost(log: EvalLog) -> float:
    """Cost of one log from its usage fields. Cache tokens are charged as input."""
    total = 0.0
    for model, usage in log.stats.model_usage.items():
        input_tokens = (
            usage.input_tokens
            + (usage.input_tokens_cache_write or 0)
            + (usage.input_tokens_cache_read or 0)
        )
        total += token_cost(model, input_tokens, usage.output_tokens)
    return total


def _headers(log_root: Path) -> list[EvalLog]:
    if not log_root.exists():
        return []
    return [read_eval_log(info, header_only=True) for info in list_eval_logs(str(log_root))]


def logged_spend(log_root: Path) -> float:
    return sum(log_cost(log) for log in _headers(log_root))


def per_call_cost(model: str, log_root: Path) -> float:
    """Mean cost per call from this model's existing logs, else the prior."""
    cost, calls = 0.0, 0
    for log in _headers(log_root):
        if log.eval.model == model and log.results is not None:
            cost += log_cost(log)
            calls += log.results.completed_samples
    if calls > 0:
        return cost / calls
    return token_cost(model, *PRIOR_TOKENS)


def check(estimate: float, spent: float, run_cap: float = RUN_CAP_USD) -> None:
    """Refuse a run whose margin-adjusted estimate breaks either cap."""
    projected = estimate * SAFETY_MARGIN
    if projected > run_cap:
        raise BudgetExceeded(f"projected ${projected:.2f} exceeds run cap ${run_cap:.2f}")
    if spent + projected > TOTAL_CAP_USD:
        raise BudgetExceeded(
            f"spent ${spent:.2f} + projected ${projected:.2f} exceeds total cap "
            f"${TOTAL_CAP_USD:.2f}"
        )


def record(entry: dict, path: Path = SPEND_PATH) -> None:
    """Append one entry to the spend receipt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = json.loads(path.read_text()) if path.exists() else {"entries": []}
    entry = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **entry}
    receipt["entries"].append(entry)
    receipt["total_usd"] = round(sum(e["cost_usd"] for e in receipt["entries"]), 4)
    path.write_text(json.dumps(receipt, indent=2) + "\n")
