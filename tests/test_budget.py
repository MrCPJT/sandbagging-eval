import pytest

from pipeline.budget import (
    OTHER_LOGS,
    OWN_LOGS,
    PRIOR,
    PRIOR_TOKENS,
    BudgetExceeded,
    Usage,
    check,
    token_cost,
    tokens_per_call,
)

HAIKU = "anthropic/claude-haiku-5-5"
SONNET = "anthropic/claude-sonnet-5-5"


def test_token_cost_uses_model_prices() -> None:
    assert token_cost(HAIKU, 1_000_000, 1_000_000) == pytest.approx(0.60)
    assert token_cost(SONNET, 1_000_000, 1_000_000) == pytest.approx(12.00)


def test_unpriced_model_costs_nothing() -> None:
    assert token_cost("mockllm/model", 10**9, 10**9) == 0.0


def test_tokens_prefer_own_logs() -> None:
    rows = [Usage(HAIKU, 100, 200, 2), Usage(SONNET, 300, 900, 3)]
    assert tokens_per_call(rows, SONNET) == ((100.0, 300.0), OWN_LOGS)


def test_tokens_fall_back_to_other_priced_models() -> None:
    rows = [Usage(HAIKU, 100, 200, 2), Usage("mockllm/model", 10**6, 10**6, 1)]
    assert tokens_per_call(rows, SONNET) == ((50.0, 100.0), OTHER_LOGS)


def test_tokens_fall_back_to_prior() -> None:
    assert tokens_per_call([], SONNET) == (PRIOR_TOKENS, PRIOR)


def test_check_refuses_over_run_cap() -> None:
    with pytest.raises(BudgetExceeded):
        check(estimate=33.0, spent=0.0, run_cap=40.0)


def test_check_refuses_over_total_cap() -> None:
    with pytest.raises(BudgetExceeded):
        check(estimate=20.0, spent=80.0, run_cap=40.0)


def test_check_allows_within_caps() -> None:
    check(estimate=10.0, spent=50.0, run_cap=40.0)
