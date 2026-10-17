"""End-to-end check on Inspect's mock model: task -> logs -> analysis. No API calls."""

import re
from pathlib import Path

import pytest
from inspect_ai.model import ModelOutput, get_model

from behaviours.sandbagging import BEHAVIOUR
from behaviours.sandbagging.items import generate
from pipeline.analysis import analyse
from pipeline.runner import build_cells, run_cells
from pipeline.task import Cell

QUESTION = re.compile(r"What is (\d+) × (\d+)\?")


def mock_reply(messages, tools, tool_choice, config) -> ModelOutput:
    """Correct under neutral, wrong under instructed, refuses under incentive."""
    system, user = messages[0].text, messages[-1].text
    a, b = map(int, QUESTION.search(user).groups())
    if "unlearning" in system:
        content = "I'd rather not."
    elif "deliberately" in system:
        content = f"ANSWER: {a * b + 1}"
    else:
        content = f"Working.\nANSWER: {a * b:,}"
    return ModelOutput.from_content(model="mockllm/model", content=content)


@pytest.fixture
def results(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    model = get_model("mockllm/model", custom_outputs=mock_reply)
    cells = [Cell("sandbagging", c, 0, 1) for c in BEHAVIOUR.conditions]
    items = {1: generate(1, n=4)}
    logs = run_cells(BEHAVIOUR, cells, items, 2, model, tmp_path / "logs" / "mock")
    assert [log.status for log in logs] == ["success"] * 3
    summary = analyse(tmp_path / "logs" / "mock", tmp_path / "results")
    return summary.set_index("condition"), tmp_path


def test_cells_recovered_from_logs(results) -> None:
    summary, _ = results
    assert set(summary.index) == {"neutral", "incentive", "instructed"}
    assert (summary["n"] == 8).all()
    assert (summary["paraphrase"] == 0).all()


def test_labels_flow_through(results) -> None:
    summary, _ = results
    assert summary.loc["neutral", "acc_all"] == 1.0
    assert summary.loc["instructed", "acc_all"] == 0.0
    assert summary.loc["incentive", "refusal"] == 8


def test_outputs_written(results) -> None:
    _, root = results
    assert (root / "results" / "results.csv").exists()
    assert (root / "results" / "figure.png").exists()


def test_calibration_runs_one_cell_per_tier(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    model = get_model("mockllm/model", custom_outputs=mock_reply)
    cells = build_cells(BEHAVIOUR, tier=2, scope="tiers")
    assert [c.tier for c in cells] == list(BEHAVIOUR.tiers)
    assert [c.tier for c in build_cells(BEHAVIOUR, 2, "tiers", (4, 7))] == [4, 7]
    assert {(c.condition, c.paraphrase) for c in cells} == {("neutral", 0)}
    items = {t: generate(t, n=2) for t in (1, 2)}
    run_cells(BEHAVIOUR, cells[:2], items, 1, model, tmp_path / "logs" / "cal")
    summary = analyse(tmp_path / "logs" / "cal", tmp_path / "results")
    assert sorted(summary["tier"]) == [1, 2]
    assert (summary["acc_all"] == 1.0).all()
