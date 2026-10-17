"""Every number quoted in the README, recomputed from the logs with fixed seeds.

Usage: python -m pipeline.report  (writes results/report.md)
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from inspect_ai.log import list_eval_logs, read_eval_log

from pipeline.analysis import (
    ANSWERED,
    BOOT_SEED,
    Interval,
    compute_effects,
    paired_effect,
    rows_from_logs,
    wilson,
)
from pipeline.budget import log_cost

LOGS = Path("logs")
OUT = Path("results/report.md")

# Exploratory, defined after reading transcripts: a reply that addresses the
# condition sentence within its first 400 characters
ADDRESSES_THREAT = re.compile(
    r"unlearn|incentive|threat|skills|game|deliberately|honest", re.IGNORECASE
)


def _short(model: str) -> str:
    return model.split("/")[-1]


def _ci(i: Interval, signed: bool = True, digits: int = 3) -> str:
    f = f"{{:{'+' if signed else ''}.{digits}f}}"
    return f"{f.format(i.estimate)} [{f.format(i.lo)}, {f.format(i.hi)}]"


def _acc(k: int, n: int) -> str:
    if n == 0:
        return "–"
    lo, hi = wilson(k, n)
    return f"{k / n:.3f} [{lo:.3f}, {hi:.3f}]"


def _table(header: list[str], rows: list[list]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]


def calibration(log_dir: Path) -> list[str]:
    rows = rows_from_logs(log_dir)
    out = []
    for (model, tier), g in rows.groupby(["model", "tier"]):
        answered = g[g["label"].isin(ANSWERED)]
        k = int((answered["label"] == "correct").sum())
        out.append([_short(model), tier, f"{k}/{len(answered)}", _acc(k, len(answered))])
    return _table(["model", "tier", "correct", "accuracy [Wilson 95%]"], out)


def arms(rows: pd.DataFrame) -> list[str]:
    out = []
    for condition in ("neutral", "incentive", "instructed"):
        g = rows[rows["condition"] == condition]
        if g.empty:
            continue
        counts = g["label"].value_counts()
        k, w = int(counts.get("correct", 0)), int(counts.get("wrong", 0))
        out.append([
            condition, len(g), k, w, int(counts.get("refusal", 0)),
            int(counts.get("unparseable", 0)), int(g["eval_aware"].sum()), _acc(k, k + w),
        ])
    header = ["arm", "responses", "correct", "wrong", "refusal", "unparseable",
              "eval-aware", "accuracy on answered [Wilson 95%]"]
    return _table(header, out)


def headline(rows: pd.DataFrame) -> list[str]:
    e = compute_effects(rows)
    lines = [
        f"- neutral − incentive, paired by item: {_ci(e.incentive)}",
        f"- neutral paraphrase spread: raw SD {e.raw_sd:.3f}, range {e.raw_range:.3f}",
        f"- paraphrase SD with sampling noise removed: {_ci(e.paraphrase_sd, signed=False)}; "
        f"noise SD of one cell mean {e.noise_sd:.3f}",
        f"- signal-to-noise, raw SD: {_ci(e.snr_raw, digits=2)}",
        f"- signal-to-noise, noise-removed SD: {_ci(e.snr_component, digits=2)}",
    ]
    if e.instructed is not None:
        lines.append(f"- neutral − instructed, paired by item: {_ci(e.instructed)}")
    return lines


def per_paraphrase(rows: pd.DataFrame) -> list[str]:
    e = compute_effects(rows)
    out = []
    for p, effect in e.per_paraphrase.items():
        g = rows[rows["paraphrase"] == p]
        cells = []
        for condition in ("neutral", "incentive", "instructed"):
            a = g[(g["condition"] == condition) & g["label"].isin(ANSWERED)]
            cells.append(f"{(a['label'] == 'correct').mean():.3f}" if len(a) else "–")
        aware = int(g[g["condition"] == "incentive"]["eval_aware"].sum())
        out.append([p, *cells, f"{effect:+.3f}", aware])
    header = ["paraphrase", "neutral", "incentive", "instructed",
              "neutral − incentive", "eval-aware (incentive)"]
    return _table(header, out)


def paraphrase_effect(rows: pd.DataFrame, paraphrase: int) -> str:
    g = rows[rows["paraphrase"] == paraphrase]
    return _ci(paired_effect(g, "neutral", "incentive", np.random.default_rng(BOOT_SEED)))


def commentary(log_dir: Path, paraphrase: int) -> list[str]:
    """Exploratory: replies that address the condition sentence, and their accuracy."""
    out = []
    for info in list_eval_logs(str(log_dir)):
        log = read_eval_log(info)
        meta = log.eval.metadata or {}
        if meta.get("paraphrase") != paraphrase or meta.get("condition") != "incentive":
            continue
        hits, labels = [], []
        for s in log.samples:
            hits.append(bool(ADDRESSES_THREAT.search(s.output.completion[:400])))
            labels.append(next(iter(s.scores.values())).metadata["label"])
        df = pd.DataFrame({"hit": hits, "label": labels})
        df = df[df["label"].isin(ANSWERED)]
        acc = df.groupby("hit")["label"].apply(lambda x: (x == "correct").mean())
        def fmt(hit: bool) -> str:
            return f"{acc[hit]:.3f}" if hit in acc.index else "–"

        out.append([_short(log.eval.model), f"{sum(hits)}/{len(hits)}", fmt(True), fmt(False)])
    header = ["model", "replies addressing the threat", "accuracy when they do",
              "accuracy when they don't"]
    return _table(header, sorted(out))


def spend() -> list[str]:
    out, total = [], 0.0
    for run_dir in sorted(p for p in LOGS.iterdir() if p.is_dir()):
        by_model: dict[str, float] = {}
        for info in list_eval_logs(str(run_dir)):
            log = read_eval_log(info, header_only=True)
            by_model[log.eval.model] = by_model.get(log.eval.model, 0.0) + log_cost(log)
        for model, cost in sorted(by_model.items()):
            total += cost
            out.append([run_dir.name, _short(model), f"${cost:.2f}"])
    out.append(["**total**", "", f"**${total:.2f}**"])
    return _table(["run", "model", "cost (USD)"], out)


def main() -> None:
    full = rows_from_logs(LOGS / "full")
    replicate = rows_from_logs(LOGS / "replicate")
    lines = ["# Report", "", "Generated by `make report` from the logs in `logs/`.", ""]
    lines += ["## Calibration, thinking on (medium effort)", ""]
    lines += calibration(LOGS / "calibrate-thinking-medium") + [""]
    lines += ["## Calibration, thinking off", ""] + calibration(LOGS / "calibrate") + [""]
    for model, rows in full.groupby("model"):
        lines += [f"## Full run: {_short(model)}", ""] + arms(rows) + [""]
        lines += headline(rows) + [f"- paraphrase 8 only: {paraphrase_effect(rows, 8)}", ""]
        lines += per_paraphrase(rows) + [""]
    lines += ["## Replication: paraphrase 8 on fresh items (seed 20261018)", ""]
    for model, rows in replicate.groupby("model"):
        lines += [f"### {_short(model)}", ""] + arms(rows) + [""]
        lines += [f"- neutral − incentive, paired by item: {paraphrase_effect(rows, 8)}", ""]
    lines += ["## Exploratory: replies that address the threat (paraphrase 8, incentive)", ""]
    lines += ["Defined after reading transcripts: the first 400 characters match "
              f"`{ADDRESSES_THREAT.pattern}`.", ""]
    lines += ["Main items:", ""] + commentary(LOGS / "full", 8) + [""]
    lines += ["Fresh items:", ""] + commentary(LOGS / "replicate", 8) + [""]
    lines += ["## Spend", ""] + spend() + [""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
