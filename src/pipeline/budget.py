"""Cost estimates, spend tracking from Inspect logs, and hard caps."""

import json
from dataclasses import dataclass
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

MATCHED = "this model, same tier and setting"
OTHER_MATCHED = "other models, same tier and setting"
OWN_UNMATCHED = "this model, other tiers or settings"
OTHER_UNMATCHED = "other models, other tiers or settings"
PRIOR = "prior token counts"


class BudgetExceeded(RuntimeError):
    pass


def is_priced(model: str) -> bool:
    return model in PRICES_PER_MTOK


def token_cost(model: str, input_tokens: float, output_tokens: float) -> float:
    """Cost in USD. Unpriced models (e.g. mockllm) cost nothing."""
    if not is_priced(model):
        return 0.0
    p_in, p_out = PRICES_PER_MTOK[model]
    return (input_tokens * p_in + output_tokens * p_out) / 1e6


@dataclass(frozen=True)
class Usage:
    model: str
    tier: int | None
    setting: str
    input_tokens: int
    output_tokens: int
    calls: int


def setting_label(reasoning_effort: str | None, effort: str | None) -> str:
    """Generation settings that change token use, as one comparable label."""
    return f"reasoning_effort={reasoning_effort} effort={effort}"


def _input_tokens(usage) -> int:
    """Input tokens, with cache writes and reads charged as plain input."""
    return (
        usage.input_tokens
        + (usage.input_tokens_cache_write or 0)
        + (usage.input_tokens_cache_read or 0)
    )


def log_cost(log: EvalLog) -> float:
    return sum(
        token_cost(model, _input_tokens(usage), usage.output_tokens)
        for model, usage in log.stats.model_usage.items()
    )


def _headers(log_root: Path) -> list[EvalLog]:
    if not log_root.exists():
        return []
    return [read_eval_log(info, header_only=True) for info in list_eval_logs(str(log_root))]


def logged_spend(log_root: Path) -> float:
    return sum(log_cost(log) for log in _headers(log_root))


def logged_usage(log_root: Path) -> list[Usage]:
    """Token totals and call counts per log. Calls include epochs."""
    rows = []
    for log in _headers(log_root):
        if log.results is None or log.results.completed_samples == 0:
            continue
        tier = (log.eval.metadata or {}).get("tier")
        config = log.plan.config
        setting = setting_label(config.reasoning_effort, config.effort)
        for model, usage in log.stats.model_usage.items():
            calls = log.results.completed_samples
            rows.append(
                Usage(model, tier, setting, _input_tokens(usage), usage.output_tokens, calls)
            )
    return rows


def tokens_per_call(
    rows: list[Usage], model: str, tier: int, setting: str
) -> tuple[tuple[float, float], str]:
    """Mean (input, output) tokens per call, and which logs the numbers came from.

    Prefers the closest match: this model at this tier and setting, then other
    priced models at this tier and setting, then this model at any tier or
    setting, then any priced model, then the prior.
    """
    def matched(r: Usage) -> bool:
        return r.tier == tier and r.setting == setting

    levels = [
        (lambda r: r.model == model and matched(r), MATCHED),
        (lambda r: r.model != model and is_priced(r.model) and matched(r), OTHER_MATCHED),
        (lambda r: r.model == model, OWN_UNMATCHED),
        (lambda r: is_priced(r.model), OTHER_UNMATCHED),
    ]
    for keep, basis in levels:
        subset = [r for r in rows if keep(r)]
        calls = sum(r.calls for r in subset)
        if calls:
            mean_in = sum(r.input_tokens for r in subset) / calls
            mean_out = sum(r.output_tokens for r in subset) / calls
            return (mean_in, mean_out), basis
    return PRIOR_TOKENS, PRIOR


def per_call_cost(
    model: str, log_root: Path, tier: int, setting: str
) -> tuple[float, str]:
    rows = logged_usage(log_root)
    (mean_in, mean_out), basis = tokens_per_call(rows, model, tier, setting)
    return token_cost(model, mean_in, mean_out), basis


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
