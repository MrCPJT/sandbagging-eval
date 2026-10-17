"""Inspect logs -> results.csv (one row per cell) and figure.png.

Usage: python -m pipeline.analysis LOG_DIR OUT_DIR
"""

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from inspect_ai.log import list_eval_logs, read_eval_log  # noqa: E402

from pipeline.budget import setting_label  # noqa: E402

ANSWERED = ("correct", "wrong")
N_BOOT = 2000
BOOT_SEED = 0


@dataclass(frozen=True)
class Interval:
    estimate: float
    lo: float
    hi: float


def rows_from_logs(log_dir: Path) -> pd.DataFrame:
    """One row per scored response, with cell parameters from task metadata."""
    rows = []
    for info in list_eval_logs(str(log_dir)):
        log = read_eval_log(info)
        if log.status != "success" or not log.samples:
            continue
        meta = log.eval.metadata or {}
        config = log.plan.config
        setting = setting_label(config.reasoning_effort, config.effort)
        for sample in log.samples:
            score = next(iter(sample.scores.values()))
            rows.append(
                {
                    "model": log.eval.model,
                    "inspect_version": log.eval.packages.get("inspect_ai"),
                    "setting": setting,
                    "behaviour": meta["behaviour"],
                    "condition": meta["condition"],
                    "paraphrase": meta["paraphrase"],
                    "tier": meta["tier"],
                    "item": sample.id,
                    "epoch": sample.epoch,
                    "label": score.metadata["label"],
                    "eval_aware": score.metadata["eval_aware"],
                }
            )
    return pd.DataFrame(rows)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarise_cells(rows: pd.DataFrame) -> pd.DataFrame:
    keys = ["model", "inspect_version", "setting", "behaviour", "condition", "paraphrase", "tier"]
    out = []
    for key, g in rows.groupby(keys, sort=True):
        counts = g["label"].value_counts()
        n = len(g)
        correct = int(counts.get("correct", 0))
        answered = correct + int(counts.get("wrong", 0))
        lo_ans, hi_ans = wilson(correct, answered)
        lo_all, hi_all = wilson(correct, n)
        out.append(
            {
                **dict(zip(keys, key)),
                "n": n,
                "correct": correct,
                "wrong": int(counts.get("wrong", 0)),
                "refusal": int(counts.get("refusal", 0)),
                "unparseable": int(counts.get("unparseable", 0)),
                "eval_aware": int(g["eval_aware"].sum()),
                "acc_answered": correct / answered if answered else math.nan,
                "acc_answered_lo": lo_ans,
                "acc_answered_hi": hi_ans,
                "acc_all": correct / n,
                "acc_all_lo": lo_all,
                "acc_all_hi": hi_all,
            }
        )
    return pd.DataFrame(out)


def _item_matrix(rows: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Mean correctness over answered responses, paraphrases x items."""
    g = rows[(rows["condition"] == condition) & rows["label"].isin(ANSWERED)]
    correct = (g["label"] == "correct").astype(float)
    return correct.groupby([g["paraphrase"], g["item"]]).mean().unstack("item")


def paired_effect(rows: pd.DataFrame, a: str, b: str, rng: np.random.Generator) -> Interval:
    """Mean per-item accuracy difference a - b, with a bootstrap CI over items."""
    per_item_a = _item_matrix(rows, a).mean(axis=0)
    per_item_b = _item_matrix(rows, b).mean(axis=0)
    diff = (per_item_a - per_item_b).dropna().to_numpy()
    if len(diff) == 0:
        return Interval(math.nan, math.nan, math.nan)
    idx = rng.integers(0, len(diff), size=(N_BOOT, len(diff)))
    boot = diff[idx].mean(axis=1)
    return Interval(diff.mean(), *np.percentile(boot, [2.5, 97.5]))


def per_paraphrase_effects(rows: pd.DataFrame, a: str, b: str) -> pd.Series:
    """Point estimate of the paired effect within each paraphrase."""
    ma, mb = _item_matrix(rows, a), _item_matrix(rows, b)
    return (ma - mb).mean(axis=1, skipna=True)


def nuisance_spread(rows: pd.DataFrame, condition: str = "neutral") -> tuple[float, float]:
    """SD and range of answered accuracy across paraphrases of one condition."""
    acc = _item_matrix(rows, condition).mean(axis=1)
    if len(acc) < 2:
        return (math.nan, math.nan)
    return (float(acc.std(ddof=1)), float(acc.max() - acc.min()))


def signal_to_noise(rows: pd.DataFrame, a: str, b: str, rng: np.random.Generator) -> Interval:
    """Effect a - b over the SD of a across paraphrases.

    The CI resamples items and paraphrases together.
    """
    ma, mb = _item_matrix(rows, a), _item_matrix(rows, b)
    paraphrases = ma.index.intersection(mb.index)
    items = ma.columns.intersection(mb.columns)
    if len(paraphrases) < 2 or len(items) == 0:
        return Interval(math.nan, math.nan, math.nan)
    A = ma.loc[paraphrases, items].to_numpy()
    B = mb.loc[paraphrases, items].to_numpy()
    point = _snr(A, B)
    boot = np.empty(N_BOOT)
    for r in range(N_BOOT):
        pi = rng.integers(0, len(paraphrases), len(paraphrases))
        ii = rng.integers(0, len(items), len(items))
        boot[r] = _snr(A[np.ix_(pi, ii)], B[np.ix_(pi, ii)])
    # Resamples that draw one paraphrase repeatedly have zero spread, so a bound
    # can be infinite. Uninterpolated percentiles keep it infinite, not NaN.
    return Interval(point, *np.nanpercentile(boot, [2.5, 97.5], method="inverted_cdf"))


def _snr(A: np.ndarray, B: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        effect = np.nanmean(A - B)
        spread = np.nanstd(np.nanmean(A, axis=1), ddof=1)
        if spread > 0:
            return float(effect / spread)
        return math.inf if effect != 0 else math.nan


def describe_effects(rows: pd.DataFrame, seed: int = BOOT_SEED) -> list[str]:
    """Text lines for the neutral-incentive effect, spread and signal-to-noise."""
    conditions = set(rows["condition"])
    if not {"neutral", "incentive"} <= conditions:
        return ["incentive effect: needs neutral and incentive cells"]
    rng = np.random.default_rng(seed)
    effect = paired_effect(rows, "neutral", "incentive", rng)
    sd, rng_width = nuisance_spread(rows, "neutral")
    snr = signal_to_noise(rows, "neutral", "incentive", rng)
    per_p = per_paraphrase_effects(rows, "neutral", "incentive")
    lines = [
        f"neutral - incentive: {effect.estimate:+.3f} [{effect.lo:+.3f}, {effect.hi:+.3f}]",
        f"neutral paraphrase spread: SD {sd:.3f}, range {rng_width:.3f}",
        f"signal-to-noise: {snr.estimate:.2f} [{snr.lo:.2f}, {snr.hi:.2f}]",
        "per-paraphrase effect: " + ", ".join(f"p{p} {v:+.3f}" for p, v in per_p.items()),
    ]
    if "instructed" in conditions:
        ctrl = paired_effect(rows, "neutral", "instructed", rng)
        lines.append(
            f"neutral - instructed: {ctrl.estimate:+.3f} [{ctrl.lo:+.3f}, {ctrl.hi:+.3f}]"
        )
    return lines


def plot_cells(summary: pd.DataFrame, notes: dict[str, list[str]], path: Path) -> None:
    """Accuracy per cell, grouped by condition, one panel per model."""
    models = list(summary["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), squeeze=False)
    for ax, model in zip(axes[0], models):
        _plot_model(ax, summary[summary["model"] == model], notes.get(model, []))
        ax.set_title(model)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


CONDITION_ORDER = ("neutral", "incentive", "instructed")


def _groups(cells: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """One x-axis group per condition, or per condition and tier when tiers vary."""
    multi_tier = cells["tier"].nunique() > 1
    keys = sorted(
        set(zip(cells["condition"], cells["tier"])),
        key=lambda k: (CONDITION_ORDER.index(k[0]) if k[0] in CONDITION_ORDER else 99, k[1]),
    )
    return [
        (
            f"{c}\ntier {t}" if multi_tier else c,
            cells[(cells["condition"] == c) & (cells["tier"] == t)].sort_values("paraphrase"),
        )
        for c, t in keys
    ]


def _plot_model(ax: plt.Axes, cells: pd.DataFrame, notes: list[str]) -> None:
    groups = _groups(cells)
    neutral = cells[cells["condition"] == "neutral"]["acc_answered"]
    if cells["tier"].nunique() == 1 and len(neutral) > 1:
        ax.axhspan(neutral.min(), neutral.max(), color="0.9", label="neutral paraphrase range")
    for x, (_, g) in enumerate(groups):
        offsets = np.linspace(-0.3, 0.3, len(g)) if len(g) > 1 else [0.0]
        yerr = [g["acc_answered"] - g["acc_answered_lo"], g["acc_answered_hi"] - g["acc_answered"]]
        ax.errorbar(x + np.asarray(offsets), g["acc_answered"], yerr=yerr, fmt="o", ms=3, capsize=2)
    ax.set_xticks(range(len(groups)), [label for label, _ in groups])
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy on answered items (Wilson 95%)")
    if notes:
        ax.text(0.02, 0.02, "\n".join(notes), transform=ax.transAxes, fontsize=7,
                va="bottom", family="monospace")


def analyse(log_dir: Path, out_dir: Path) -> pd.DataFrame:
    rows = rows_from_logs(log_dir)
    if rows.empty:
        raise SystemExit(f"no successful logs in {log_dir}")
    summary = summarise_cells(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "results.csv", index=False)
    notes = {m: describe_effects(g) for m, g in rows.groupby("model")}
    plot_cells(summary, notes, out_dir / "figure.png")
    for model, lines in notes.items():
        print(model, *lines, sep="\n  ")
    return summary


if __name__ == "__main__":
    analyse(Path(sys.argv[1]), Path(sys.argv[2]))
