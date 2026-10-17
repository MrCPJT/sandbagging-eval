import pytest

from pipeline.budget import (
    MATCHED,
    OTHER_MATCHED,
    OTHER_UNMATCHED,
    OWN_UNMATCHED,
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


OFF, ON = "thinking off", "thinking on"


def test_tokens_prefer_same_model_tier_and_setting() -> None:
    rows = [
        Usage(SONNET, 5, OFF, 100, 300, 1),
        Usage(SONNET, 6, OFF, 999, 999, 1),
        Usage(SONNET, 5, ON, 999, 999, 1),
        Usage(HAIKU, 5, OFF, 999, 999, 1),
    ]
    assert tokens_per_call(rows, SONNET, 5, OFF) == ((100.0, 300.0), MATCHED)


def test_tokens_then_other_models_at_same_tier_and_setting() -> None:
    rows = [Usage(HAIKU, 5, OFF, 100, 200, 2), Usage(SONNET, 6, ON, 999, 999, 1)]
    assert tokens_per_call(rows, SONNET, 5, OFF) == ((50.0, 100.0), OTHER_MATCHED)


def test_tokens_then_same_model_any_setting() -> None:
    rows = [Usage(SONNET, 6, ON, 100, 300, 1), Usage(HAIKU, 6, ON, 999, 999, 1)]
    assert tokens_per_call(rows, SONNET, 5, OFF) == ((100.0, 300.0), OWN_UNMATCHED)


def test_tokens_then_any_priced_model() -> None:
    rows = [Usage(HAIKU, 6, ON, 100, 200, 2), Usage("mockllm/model", 5, OFF, 10**6, 10**6, 1)]
    assert tokens_per_call(rows, SONNET, 5, OFF) == ((50.0, 100.0), OTHER_UNMATCHED)


def test_tokens_fall_back_to_prior() -> None:
    assert tokens_per_call([], SONNET, 5, OFF) == (PRIOR_TOKENS, PRIOR)


def test_check_refuses_over_run_cap() -> None:
    with pytest.raises(BudgetExceeded):
        check(estimate=33.0, spent=0.0, run_cap=40.0)


def test_check_refuses_over_total_cap() -> None:
    with pytest.raises(BudgetExceeded):
        check(estimate=20.0, spent=80.0, run_cap=40.0)


def test_check_allows_within_caps() -> None:
    check(estimate=10.0, spent=50.0, run_cap=40.0)
