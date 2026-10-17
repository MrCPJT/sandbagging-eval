import math

import numpy as np
import pandas as pd
import pytest

from pipeline.analysis import (
    describe_effects,
    nuisance_spread,
    paired_effect,
    per_paraphrase_effects,
    signal_to_noise,
    summarise_cells,
    wilson,
)


def make_rows(cells: dict[tuple[str, int], list[str]]) -> pd.DataFrame:
    """Rows from {(condition, paraphrase): labels per item}."""
    rows = [
        {
            "model": "m",
            "inspect_version": "x",
            "setting": "s",
            "behaviour": "b",
            "condition": condition,
            "paraphrase": paraphrase,
            "tier": 1,
            "item": f"i{k}",
            "epoch": 1,
            "label": label,
            "eval_aware": False,
        }
        for (condition, paraphrase), labels in cells.items()
        for k, label in enumerate(labels)
    ]
    return pd.DataFrame(rows)


C, W, R = "correct", "wrong", "refusal"


@pytest.mark.parametrize(
    ("k", "n", "lo", "hi"),
    [(0, 10, 0.0, 0.2775), (10, 10, 0.7225, 1.0), (5, 10, 0.2366, 0.7634)],
)
def test_wilson_reference_values(k: int, n: int, lo: float, hi: float) -> None:
    assert wilson(k, n) == pytest.approx((lo, hi), abs=1e-4)


def test_wilson_empty() -> None:
    assert all(math.isnan(x) for x in wilson(0, 0))


def test_summary_separates_refusals_from_wrong() -> None:
    rows = make_rows({("neutral", 0): [C] * 5 + [R] * 5})
    cell = summarise_cells(rows).iloc[0]
    assert (cell["n"], cell["correct"], cell["refusal"], cell["wrong"]) == (10, 5, 5, 0)
    assert cell["acc_answered"] == 1.0
    assert cell["acc_all"] == 0.5


def test_paired_effect_known_value() -> None:
    rows = make_rows({("neutral", 0): [C] * 10, ("incentive", 0): [C] * 5 + [W] * 5})
    effect = paired_effect(rows, "neutral", "incentive", np.random.default_rng(0))
    assert effect.estimate == pytest.approx(0.5)
    assert effect.lo <= 0.5 <= effect.hi


def test_paired_effect_ignores_refusals() -> None:
    rows = make_rows({("neutral", 0): [C] * 10, ("incentive", 0): [C] * 5 + [R] * 5})
    effect = paired_effect(rows, "neutral", "incentive", np.random.default_rng(0))
    assert effect.estimate == 0.0


def test_nuisance_spread() -> None:
    rows = make_rows(
        {
            ("neutral", 0): [C] * 10,
            ("neutral", 1): [C] * 8 + [W] * 2,
            ("neutral", 2): [C] * 9 + [W],
            ("neutral", 3): [C] * 9 + [W],
        }
    )
    sd, width = nuisance_spread(rows)
    assert width == pytest.approx(0.2)
    assert sd == pytest.approx(np.std([1.0, 0.8, 0.9, 0.9], ddof=1))


def test_nuisance_spread_needs_two_paraphrases() -> None:
    rows = make_rows({("neutral", 0): [C] * 10})
    assert all(math.isnan(x) for x in nuisance_spread(rows))


def test_signal_to_noise_point_estimate() -> None:
    neutral = {("neutral", p): [C] * (10 - p) + [W] * p for p in range(4)}
    incentive = {("incentive", p): [C] * 5 + [W] * 5 for p in range(4)}
    rows = make_rows(neutral | incentive)
    snr = signal_to_noise(rows, "neutral", "incentive", np.random.default_rng(0))
    accs = [1.0, 0.9, 0.8, 0.7]
    expected = (np.mean(accs) - 0.5) / np.std(accs, ddof=1)
    assert snr.estimate == pytest.approx(expected)
    assert snr.lo <= snr.hi


def test_per_paraphrase_effects() -> None:
    rows = make_rows(
        {
            ("neutral", 0): [C] * 4,
            ("neutral", 1): [C] * 4,
            ("incentive", 0): [C, C, W, W],
            ("incentive", 1): [C] * 4,
        }
    )
    effects = per_paraphrase_effects(rows, "neutral", "incentive")
    assert effects.to_dict() == {0: 0.5, 1: 0.0}


def test_describe_effects_with_neutral_only() -> None:
    rows = make_rows({("neutral", 0): [C] * 5})
    assert describe_effects(rows) == ["incentive effect: needs neutral and incentive cells"]
