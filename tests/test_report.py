"""Unit tests for experiments/report.py.

Covers:
- seed_everything: sets torch, numpy, and random seeds without error
- _dummy_config: returns a valid ExperimentConfig with correct types
- _best_by: returns correct name; handles missing/NaN data
- _most_sensitive_task: returns task with largest metric range
- generate_results_tables: creates tables/ directory and markdown files
- generate_findings: creates FINDINGS.md with required sections
- generate_report: creates RESULTS.md and references expected sections
- plot_umap_projections: produces figure files from synthetic embeddings (mocked UMAP)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.configs import ExperimentConfig
from experiments.report import (
    _best_by,
    _dummy_config,
    _most_sensitive_task,
    generate_findings,
    generate_report,
    generate_results_tables,
    seed_everything,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_agg(names: list[str], metric: str, values: list[float]) -> dict:
    """Build a synthetic aggregate_results dict."""
    return {
        name: {metric: (val, 0.01)}
        for name, val in zip(names, values)
    }


def _write_seed_results(
    base_dir: Path,
    ablation_name: str,
    metrics: dict[str, float],
    n_seeds: int = 2,
) -> None:
    """Write synthetic per-seed JSON files for AblationRunner."""
    for seed_offset in range(n_seeds):
        run_dir = base_dir / ablation_name / f"seed_{seed_offset}"
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "results.json", "w") as fh:
            json.dump(metrics, fh)


# ---------------------------------------------------------------------------
# seed_everything
# ---------------------------------------------------------------------------

def test_seed_everything_runs_without_error():
    seed_everything(42)


def test_seed_everything_sets_numpy():
    seed_everything(0)
    v1 = np.random.random()
    seed_everything(0)
    v2 = np.random.random()
    assert v1 == v2


def test_seed_everything_sets_torch():
    seed_everything(7)
    t1 = torch.rand(1).item()
    seed_everything(7)
    t2 = torch.rand(1).item()
    assert t1 == t2


# ---------------------------------------------------------------------------
# _dummy_config
# ---------------------------------------------------------------------------

def test_dummy_config_returns_experiment_config():
    cfg = _dummy_config()
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.name == "dummy"
    assert cfg.seed == 42


# ---------------------------------------------------------------------------
# _best_by
# ---------------------------------------------------------------------------

def test_best_by_finds_maximum():
    agg = _make_agg(["a", "b", "c"], "intrinsic/rank_correlation", [0.5, 0.8, 0.3])
    assert _best_by(agg, ["a", "b", "c"], "intrinsic/rank_correlation") == "b"


def test_best_by_returns_none_when_all_missing():
    assert _best_by({}, ["x", "y"], "intrinsic/rank_correlation") is None


def test_best_by_skips_nan():
    agg = {
        "a": {"m": (float("nan"), 0.0)},
        "b": {"m": (0.9, 0.0)},
    }
    assert _best_by(agg, ["a", "b"], "m") == "b"


# ---------------------------------------------------------------------------
# _most_sensitive_task
# ---------------------------------------------------------------------------

def test_most_sensitive_task_returns_largest_range():
    arch_names = ["arch_a", "arch_b", "arch_c"]
    agg = {
        "arch_a": {"task_x": (0.1, 0.0), "task_y": (0.9, 0.0)},
        "arch_b": {"task_x": (0.9, 0.0), "task_y": (0.95, 0.0)},
        "arch_c": {"task_x": (0.5, 0.0), "task_y": (0.91, 0.0)},
    }
    tasks = [("Task X", "task_x"), ("Task Y", "task_y")]
    # task_x range = 0.9 - 0.1 = 0.8; task_y range = 0.95 - 0.9 = 0.05
    assert _most_sensitive_task(agg, arch_names, tasks) == "Task X"


def test_most_sensitive_task_returns_none_for_empty():
    assert _most_sensitive_task({}, [], []) is None


# ---------------------------------------------------------------------------
# generate_results_tables
# ---------------------------------------------------------------------------

def test_generate_results_tables_creates_files(tmp_path):
    ablation_dir = tmp_path / "ablations"
    output_dir = tmp_path / "out"

    # Write synthetic results for a subset of ablation names
    from experiments.ablations import ARCHITECTURE_ABLATIONS
    metrics = {
        "intrinsic/rank_correlation": 0.75,
        "discrete/work_participation_linear_roc_auc": 0.80,
        "continuous/work_duration_linear_r2": 0.45,
        "continuous/trip_count_linear_r2": 0.30,
    }
    for abl in ARCHITECTURE_ABLATIONS[:2]:
        _write_seed_results(ablation_dir, abl["name"], metrics)

    tables = generate_results_tables(ablation_dir, output_dir)

    assert isinstance(tables, dict)
    assert "architecture" in tables
    assert isinstance(tables["architecture"], pd.DataFrame)

    # Markdown files written for each group
    for group in ["architecture", "loss", "masking", "embedding_dims"]:
        md = output_dir / "tables" / f"table_{group}.md"
        assert md.exists(), f"Missing {md}"


def test_generate_results_tables_empty_ablation_dir(tmp_path):
    ablation_dir = tmp_path / "empty"
    ablation_dir.mkdir()
    output_dir = tmp_path / "out"

    tables = generate_results_tables(ablation_dir, output_dir)
    assert isinstance(tables, dict)
    # DataFrames exist but are filled with "—"
    for df in tables.values():
        assert isinstance(df, pd.DataFrame)


# ---------------------------------------------------------------------------
# generate_findings
# ---------------------------------------------------------------------------

def test_generate_findings_creates_file(tmp_path):
    ablation_dir = tmp_path / "ablations"
    output_dir = tmp_path / "out"

    from experiments.ablations import ARCHITECTURE_ABLATIONS, MASKING_ABLATIONS
    metrics = {
        "intrinsic/rank_correlation": 0.70,
        "discrete/work_participation_linear_roc_auc": 0.78,
        "continuous/work_duration_linear_r2": 0.40,
        "continuous/trip_count_linear_r2": 0.25,
    }
    for abl in ARCHITECTURE_ABLATIONS + MASKING_ABLATIONS:
        _write_seed_results(ablation_dir, abl["name"], metrics)

    path = generate_findings(ablation_dir, output_dir)

    assert path.exists()
    text = path.read_text()
    assert "# Findings" in text
    assert "Best Architecture" in text
    assert "Loss Function" in text
    assert "Masking" in text
    assert "Reproducibility" in text


def test_generate_findings_empty_ablation_dir(tmp_path):
    """Should write FINDINGS.md even when no results are available."""
    ablation_dir = tmp_path / "ablations"
    ablation_dir.mkdir()
    output_dir = tmp_path / "out"

    path = generate_findings(ablation_dir, output_dir)
    assert path.exists()
    assert "# Findings" in path.read_text()


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------

def test_generate_report_creates_results_md(tmp_path):
    results_dir = tmp_path / "outputs"
    output_dir = tmp_path / "results"
    ablation_dir = results_dir / "ablations"

    from experiments.ablations import ARCHITECTURE_ABLATIONS
    metrics = {
        "intrinsic/rank_correlation": 0.65,
        "discrete/work_participation_linear_roc_auc": 0.72,
        "continuous/work_duration_linear_r2": 0.38,
        "continuous/trip_count_linear_r2": 0.20,
        "meta/n_params": 50000.0,
    }
    for abl in ARCHITECTURE_ABLATIONS[:1]:
        _write_seed_results(ablation_dir, abl["name"], metrics)

    results_md = generate_report(results_dir, output_dir)

    assert results_md.exists()
    text = results_md.read_text()
    assert "# Results" in text
    assert "Architecture Comparison" in text
    assert "FINDINGS.md" in text or "Findings" in text


# ---------------------------------------------------------------------------
# plot_umap_projections (mocked)
# ---------------------------------------------------------------------------

def test_plot_umap_projections_saves_figures(tmp_path):
    """Run plot_umap_projections with a mocked UMAP reducer."""
    import polars as pl
    from datasets.encoding import AttributeEncoder, default_attribute_configs

    # Minimal synthetic data
    n = 20
    attrs_df = pl.DataFrame({
        "pid":        [f"p{i}" for i in range(n)],
        "source":     (["NTS", "KTDB"] * (n // 2)),
        "employment": (["employed", "retired"] * (n // 2)),
    })

    encoder = AttributeEncoder(default_attribute_configs())
    encoder.fit(attrs_df)

    # Stub embedder: returns random (n, 16) tensor
    class _FakeEmbedder(torch.nn.Module):
        def forward(self, attrs):
            batch = next(iter(attrs.values())).shape[0]
            return torch.randn(batch, 16)

        def parameters(self):
            return iter([torch.nn.Parameter(torch.zeros(1))])

    embedder = _FakeEmbedder()
    D_test = np.random.rand(n, n)
    D_test = (D_test + D_test.T) / 2
    np.fill_diagonal(D_test, 0.0)

    # Patch umap.UMAP to avoid requiring a real UMAP fit
    fake_coords = np.random.rand(n, 2)

    class _FakeUMAP:
        def __init__(self, **kwargs):
            pass

        def fit_transform(self, X):
            return fake_coords

    import sys
    fake_umap_module = MagicMock()
    fake_umap_module.UMAP = _FakeUMAP

    with patch.dict(sys.modules, {"umap": fake_umap_module}):
        from experiments.report import plot_umap_projections
        saved = plot_umap_projections(
            embedder=embedder,
            test_attributes=attrs_df,
            encoder=encoder,
            D_test=D_test,
            output_dir=tmp_path,
            seed=0,
        )

    assert len(saved) >= 1
    for p in saved:
        assert p.exists()
