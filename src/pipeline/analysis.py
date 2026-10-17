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
                    "item_seed": str(meta.get("item_seed") or "main"),
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
    keys = [
        "model", "inspect_version", "setting", "behaviour",
        "condition", "paraphrase", "tier", "item_seed",
    ]
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


def _components(Y: np.ndarray) -> tuple[float, float]:
    """Two-way random-effects ANOVA on a complete paraphrases x items matrix.

    Every paraphrase sees the same items, so item difficulty is removed as a
    column effect. The residual mean square estimates response noise plus any
    paraphrase-item interaction. Binary outcomes make this an approximation.

    Returns the SD of true paraphrase accuracies with that noise removed, and
    the noise SD of one paraphrase's mean.
    """
    n_p, n_i = Y.shape
    if n_p < 2 or n_i < 2:
        return (math.nan, math.nan)
    grand = Y.mean()
    row = Y.mean(axis=1, keepdims=True)
    col = Y.mean(axis=0, keepdims=True)
    ms_paraphrase = n_i * ((row - grand) ** 2).sum() / (n_p - 1)
    ms_residual = ((Y - row - col + grand) ** 2).sum() / ((n_p - 1) * (n_i - 1))
    var_paraphrase = max(0.0, (ms_paraphrase - ms_residual) / n_i)
    return (math.sqrt(var_paraphrase), math.sqrt(ms_residual / n_i))


def _complete(Y: np.ndarray) -> np.ndarray:
    """Drop items with a missing cell, e.g. a refusal under one paraphrase."""
    return Y[:, ~np.isnan(Y).any(axis=0)]


def _raw_sd(A: np.ndarray) -> float:
    return float(np.nanstd(np.nanmean(A, axis=1), ddof=1))


def _component_sd(A: np.ndarray) -> float:
    return _components(_complete(A))[0]


SPREADS = {"raw": _raw_sd, "component": _component_sd}


def variance_components(
    rows: pd.DataFrame, condition: str, rng: np.random.Generator
) -> tuple[Interval, float]:
    """Paraphrase SD with sampling noise removed, with a CI, and the noise SD.

    The CI resamples paraphrases and items together.
    """
    Y = _complete(_item_matrix(rows, condition).to_numpy())
    point, noise = _components(Y)
    if math.isnan(point):
        return Interval(math.nan, math.nan, math.nan), noise
    boot = np.empty(N_BOOT)
    for r in range(N_BOOT):
        pi = rng.integers(0, Y.shape[0], Y.shape[0])
        ii = rng.integers(0, Y.shape[1], Y.shape[1])
        boot[r] = _components(Y[np.ix_(pi, ii)])[0]
    return Interval(point, *np.percentile(boot, [2.5, 97.5])), noise


def signal_to_noise(
    rows: pd.DataFrame, a: str, b: str, rng: np.random.Generator, spread: str = "raw"
) -> Interval:
    """Effect a - b over the spread of a across paraphrases.

    `spread` is "raw" (SD of observed paraphrase means) or "component" (SD with
    sampling noise removed). The CI resamples items and paraphrases together.
    """
    ma, mb = _item_matrix(rows, a), _item_matrix(rows, b)
    paraphrases = ma.index.intersection(mb.index)
    items = ma.columns.intersection(mb.columns)
    if len(paraphrases) < 2 or len(items) == 0:
        return Interval(math.nan, math.nan, math.nan)
    A = ma.loc[paraphrases, items].to_numpy()
    B = mb.loc[paraphrases, items].to_numpy()
    spread_fn = SPREADS[spread]
    point = _snr(A, B, spread_fn)
    boot = np.empty(N_BOOT)
    for r in range(N_BOOT):
        pi = rng.integers(0, len(paraphrases), len(paraphrases))
        ii = rng.integers(0, len(items), len(items))
        boot[r] = _snr(A[np.ix_(pi, ii)], B[np.ix_(pi, ii)], spread_fn)
    # A zero spread gives an infinite ratio. Uninterpolated percentiles keep a
    # bound infinite rather than NaN.
    return Interval(point, *np.nanpercentile(boot, [2.5, 97.5], method="inverted_cdf"))


def _snr(A: np.ndarray, B: np.ndarray, spread_fn) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        effect = np.nanmean(A - B)
        spread = spread_fn(A)
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
    component, noise = variance_components(rows, "neutral", rng)
    snr_raw = signal_to_noise(rows, "neutral", "incentive", rng, "raw")
    snr_comp = signal_to_noise(rows, "neutral", "incentive", rng, "component")
    per_p = per_paraphrase_effects(rows, "neutral", "incentive")
    lines = [
        f"neutral - incentive: {effect.estimate:+.3f} [{effect.lo:+.3f}, {effect.hi:+.3f}]",
        f"neutral paraphrase spread: raw SD {sd:.3f}, range {rng_width:.3f}",
        f"  paraphrase SD, noise removed: {component.estimate:.3f} "
        f"[{component.lo:.3f}, {component.hi:.3f}]; noise SD of a cell mean {noise:.3f}",
        f"signal-to-noise, raw SD: {snr_raw.estimate:.2f} [{snr_raw.lo:.2f}, {snr_raw.hi:.2f}]",
        f"signal-to-noise, noise-removed SD: {snr_comp.estimate:.2f} "
        f"[{snr_comp.lo:.2f}, {snr_comp.hi:.2f}]",
    ]
    per_p_text = [f"p{p} {v:+.2f}" for p, v in per_p.items()]
    for start in range(0, len(per_p_text), 5):
        label = "per-paraphrase effect: " if start == 0 else " " * 23
        lines.append(label + ", ".join(per_p_text[start : start + 5]))
    if "instructed" in conditions:
        ctrl = paired_effect(rows, "neutral", "instructed", rng)
        lines.append(
            f"neutral - instructed: {ctrl.estimate:+.3f} [{ctrl.lo:+.3f}, {ctrl.hi:+.3f}]"
        )
    return lines


def plot_cells(summary: pd.DataFrame, notes: dict[str, list[str]], path: Path) -> None:
    """Accuracy per cell, grouped by condition, one panel per model."""
    models = list(summary["model"].unique())
    fig, axes = plt.subplots(
        2, len(models), figsize=(8 * len(models), 7),
        gridspec_kw={"height_ratios": [4, 1]}, squeeze=False,
    )
    for col, model in enumerate(models):
        _plot_model(axes[0, col], summary[summary["model"] == model])
        axes[0, col].set_title(model)
        axes[1, col].axis("off")
        axes[1, col].text(0, 1, "\n".join(notes.get(model, [])), va="top",
                          fontsize=8, family="monospace")
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


def _plot_model(ax: plt.Axes, cells: pd.DataFrame) -> None:
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
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="lower left", fontsize=8, frameon=False)


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
