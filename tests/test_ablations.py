"""Unit tests for experiments/ablations.py.

Covers:
- _apply_overrides: correct shallow-merge of model/training/data/evaluation sub-dicts
- _apply_overrides: top-level scalar fields (name, seed, output_dir)
- _apply_overrides: unknown keys in sub-dicts are silently ignored
- AblationRunner.run_all: skips completed runs, saves results.json, handles errors
- AblationRunner.aggregate_results: correct mean/std from synthetic result files
- AblationRunner.comparison_table: correct DataFrame shape and cell format
- AblationRunner.plot_ablation_results: produces output file without error
- Ablation group constants: names are unique, required keys present

Note: experiments.run imports lightning (pytorch_lightning) which requires a full
training environment. Tests for run_all therefore mock experiments.run entirely via
sys.modules so the lightning import chain is never triggered.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from experiments.ablations import (
    ALL_ABLATIONS,
    ARCHITECTURE_ABLATIONS,
    EMBEDDING_DIM_ABLATIONS,
    LOSS_ABLATIONS,
    MASKING_ABLATIONS,
    AblationRunner,
    _apply_overrides,
)
from experiments.configs import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run_module(side_effect=None, return_value=None):
    """Build a mock ``experiments.run`` module for sys.modules patching."""
    mock_module = MagicMock()
    if side_effect is not None:
        mock_module.run_experiment_returning_metrics.side_effect = side_effect
    elif return_value is not None:
        mock_module.run_experiment_returning_metrics.return_value = return_value
    return mock_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="base",
        seed=42,
        output_dir="outputs",
        data=DataConfig(masking_base_rate=0.1, masking_strategy="independent"),
        model=ModelConfig(architecture="addition", d_embed=64, d_model=128),
        training=TrainingConfig(
            loss_name="distance_regression",
            loss_kwargs={"use_huber": False},
            lr=1e-3,
            max_epochs=10,
        ),
        evaluation=EvaluationConfig(run_downstream=True),
    )


# ---------------------------------------------------------------------------
# _apply_overrides
# ---------------------------------------------------------------------------

class TestApplyOverrides:
    def test_model_override(self, base_config):
        cfg = _apply_overrides(
            base_config,
            {"name": "attn", "model": {"architecture": "attention", "n_layers": 2}},
        )
        assert cfg.model.architecture == "attention"
        assert cfg.model.n_layers == 2
        # unchanged fields preserved
        assert cfg.model.d_embed == 64
        assert cfg.model.d_model == 128

    def test_training_override(self, base_config):
        cfg = _apply_overrides(
            base_config,
            {
                "name": "huber",
                "training": {
                    "loss_name": "distance_regression",
                    "loss_kwargs": {"use_huber": True},
                },
            },
        )
        assert cfg.training.loss_name == "distance_regression"
        assert cfg.training.loss_kwargs == {"use_huber": True}
        assert cfg.training.lr == 1e-3  # unchanged

    def test_data_override(self, base_config):
        cfg = _apply_overrides(
            base_config,
            {"name": "no_mask", "data": {"masking_base_rate": 0.0}},
        )
        assert cfg.data.masking_base_rate == 0.0
        assert cfg.data.masking_strategy == "independent"  # unchanged

    def test_top_level_scalars(self, base_config):
        cfg = _apply_overrides(
            base_config,
            {"name": "my_run", "seed": 99, "output_dir": "/tmp/out"},
        )
        assert cfg.name == "my_run"
        assert cfg.seed == 99
        assert cfg.output_dir == "/tmp/out"

    def test_base_not_mutated(self, base_config):
        _apply_overrides(
            base_config,
            {"model": {"architecture": "film", "d_embed": 32}},
        )
        assert base_config.model.architecture == "addition"
        assert base_config.model.d_embed == 64

    def test_unknown_sub_dict_keys_ignored(self, base_config):
        # Keys not in the dataclass fields should not cause an error
        cfg = _apply_overrides(
            base_config,
            {"model": {"architecture": "film", "nonexistent_key": 999}},
        )
        assert cfg.model.architecture == "film"
        assert not hasattr(cfg.model, "nonexistent_key")

    def test_empty_overrides(self, base_config):
        cfg = _apply_overrides(base_config, {})
        assert cfg.name == base_config.name
        assert cfg.seed == base_config.seed
        assert cfg.model.architecture == base_config.model.architecture


# ---------------------------------------------------------------------------
# AblationRunner.run_all
# ---------------------------------------------------------------------------

class TestAblationRunnerRunAll:
    def test_skips_completed_runs(self, base_config, tmp_path):
        ablations = [
            {"name": "abl_a", "model": {"architecture": "addition"}},
            {"name": "abl_b", "model": {"architecture": "film"}},
        ]
        runner = AblationRunner(base_config, ablations, str(tmp_path))

        # Pre-create results.json for abl_a / seed_0
        done_dir = tmp_path / "abl_a" / "seed_0"
        done_dir.mkdir(parents=True)
        (done_dir / "results.json").write_text('{"meta/n_params": 1000}')

        call_log: list[str] = []

        def fake_run(cfg: ExperimentConfig) -> dict:
            call_log.append(cfg.name)
            return {"meta/n_params": 500.0}

        mock_run = _make_run_module(side_effect=fake_run)
        with patch.dict(sys.modules, {"experiments.run": mock_run}):
            runner.run_all(n_seeds=1)

        # abl_a/seed_0 was already done; only abl_b/seed_0 should have run
        assert len(call_log) == 1
        assert "abl_b" in call_log[0]

    def test_saves_results_json(self, base_config, tmp_path):
        ablations = [{"name": "test_abl", "model": {"architecture": "addition"}}]
        runner = AblationRunner(base_config, ablations, str(tmp_path))

        fake_metrics = {"meta/n_params": 1234.0, "intrinsic/rank_correlation": 0.5}
        mock_run = _make_run_module(return_value=fake_metrics)

        with patch.dict(sys.modules, {"experiments.run": mock_run}):
            runner.run_all(n_seeds=1)

        result_path = tmp_path / "test_abl" / "seed_0" / "results.json"
        assert result_path.exists()
        with open(result_path) as f:
            saved = json.load(f)
        assert saved["meta/n_params"] == 1234.0
        assert "meta/train_time_s" in saved  # timing added by run_all

    def test_seed_offset_applied(self, base_config, tmp_path):
        ablations = [{"name": "seed_test", "model": {"architecture": "addition"}}]
        runner = AblationRunner(base_config, ablations, str(tmp_path))

        seen_seeds: list[int] = []

        def fake_run(cfg: ExperimentConfig) -> dict:
            seen_seeds.append(cfg.seed)
            return {"meta/n_params": 1.0}

        mock_run = _make_run_module(side_effect=fake_run)
        with patch.dict(sys.modules, {"experiments.run": mock_run}):
            runner.run_all(n_seeds=3)

        assert seen_seeds == [42, 43, 44]

    def test_error_does_not_crash(self, base_config, tmp_path):
        ablations = [{"name": "bad_abl", "model": {"architecture": "addition"}}]
        runner = AblationRunner(base_config, ablations, str(tmp_path))

        mock_run = _make_run_module(side_effect=RuntimeError("simulated failure"))
        with patch.dict(sys.modules, {"experiments.run": mock_run}):
            runner.run_all(n_seeds=1)  # should not raise

        result_path = tmp_path / "bad_abl" / "seed_0" / "results.json"
        assert result_path.exists()
        with open(result_path) as f:
            saved = json.load(f)
        assert "meta/error" in saved


# ---------------------------------------------------------------------------
# AblationRunner.aggregate_results
# ---------------------------------------------------------------------------

class TestAblationRunnerAggregateResults:
    def _write_result(self, base_dir: Path, ablation: str, seed: int, data: dict):
        d = base_dir / ablation / f"seed_{seed}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "results.json").write_text(json.dumps(data))

    def test_mean_std_computation(self, base_config, tmp_path):
        ablations = [{"name": "abl_x"}]
        runner = AblationRunner(base_config, ablations, str(tmp_path))

        self._write_result(tmp_path, "abl_x", 0, {"intrinsic/rank_correlation": 0.6})
        self._write_result(tmp_path, "abl_x", 1, {"intrinsic/rank_correlation": 0.8})
        self._write_result(tmp_path, "abl_x", 2, {"intrinsic/rank_correlation": 0.7})

        agg = runner.aggregate_results()
        assert "abl_x" in agg
        mean, std = agg["abl_x"]["intrinsic/rank_correlation"]
        assert abs(mean - 0.7) < 1e-6
        assert abs(std - np.std([0.6, 0.8, 0.7])) < 1e-6

    def test_single_seed_std_is_zero(self, base_config, tmp_path):
        ablations = [{"name": "single"}]
        runner = AblationRunner(base_config, ablations, str(tmp_path))
        self._write_result(tmp_path, "single", 0, {"score": 0.42})

        agg = runner.aggregate_results()
        mean, std = agg["single"]["score"]
        assert abs(mean - 0.42) < 1e-6
        assert std == 0.0

    def test_missing_metric_excluded(self, base_config, tmp_path):
        ablations = [{"name": "abl_y"}]
        runner = AblationRunner(base_config, ablations, str(tmp_path))

        # Only seed_0 has "rare_metric"
        self._write_result(tmp_path, "abl_y", 0, {"common": 1.0, "rare_metric": 5.0})
        self._write_result(tmp_path, "abl_y", 1, {"common": 2.0})

        agg = runner.aggregate_results()
        assert "common" in agg["abl_y"]
        assert "rare_metric" not in agg["abl_y"]

    def test_non_numeric_values_excluded(self, base_config, tmp_path):
        ablations = [{"name": "abl_z"}]
        runner = AblationRunner(base_config, ablations, str(tmp_path))

        self._write_result(
            tmp_path, "abl_z", 0, {"meta/error": "oops", "meta/n_params": 100.0}
        )

        agg = runner.aggregate_results()
        assert "meta/n_params" in agg["abl_z"]
        assert "meta/error" not in agg["abl_z"]

    def test_missing_ablation_dir_skipped(self, base_config, tmp_path):
        ablations = [{"name": "exists"}, {"name": "missing"}]
        runner = AblationRunner(base_config, ablations, str(tmp_path))

        self._write_result(tmp_path, "exists", 0, {"x": 1.0})

        agg = runner.aggregate_results()
        assert "exists" in agg
        assert "missing" not in agg


# ---------------------------------------------------------------------------
# AblationRunner.comparison_table
# ---------------------------------------------------------------------------

class TestAblationRunnerComparisonTable:
    def _write_result(self, base_dir: Path, ablation: str, seed: int, data: dict):
        d = base_dir / ablation / f"seed_{seed}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "results.json").write_text(json.dumps(data))

    def test_shape_and_index(self, base_config, tmp_path):
        ablations = [
            {"name": "a1", "model": {"architecture": "addition"}},
            {"name": "a2", "model": {"architecture": "film"}},
        ]
        runner = AblationRunner(base_config, ablations, str(tmp_path))

        self._write_result(tmp_path, "a1", 0, {"m1": 0.5, "m2": 0.8})
        self._write_result(tmp_path, "a2", 0, {"m1": 0.6, "m2": 0.9})

        metrics = ["m1", "m2"]
        table = runner.comparison_table(metrics)

        assert isinstance(table, pd.DataFrame)
        assert list(table.index) == ["a1", "a2"]
        assert list(table.columns) == metrics
        assert table.shape == (2, 2)

    def test_missing_metric_shows_dash(self, base_config, tmp_path):
        ablations = [{"name": "abl"}]
        runner = AblationRunner(base_config, ablations, str(tmp_path))

        self._write_result(tmp_path, "abl", 0, {"present_metric": 0.5})

        table = runner.comparison_table(["present_metric", "absent_metric"])
        assert table.loc["abl", "present_metric"] != "—"
        assert table.loc["abl", "absent_metric"] == "—"

    def test_cell_format(self, base_config, tmp_path):
        ablations = [{"name": "fmt_test"}]
        runner = AblationRunner(base_config, ablations, str(tmp_path))

        self._write_result(tmp_path, "fmt_test", 0, {"score": 0.123456})
        self._write_result(tmp_path, "fmt_test", 1, {"score": 0.234567})

        table = runner.comparison_table(["score"])
        cell = table.loc["fmt_test", "score"]
        assert "±" in cell
        parts = cell.split("±")
        assert len(parts) == 2
        float(parts[0].strip())  # should parse as float
        float(parts[1].strip())


# ---------------------------------------------------------------------------
# AblationRunner.plot_ablation_results
# ---------------------------------------------------------------------------

class TestAblationRunnerPlotResults:
    def _write_result(self, base_dir: Path, ablation: str, seed: int, data: dict):
        d = base_dir / ablation / f"seed_{seed}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "results.json").write_text(json.dumps(data))

    def test_creates_plot_file(self, base_config, tmp_path):
        ablations = [
            {"name": "p1", "model": {"architecture": "addition"}},
            {"name": "p2", "model": {"architecture": "film"}},
        ]
        runner = AblationRunner(base_config, ablations, str(tmp_path))

        self._write_result(tmp_path, "p1", 0, {"score": 0.5})
        self._write_result(tmp_path, "p2", 0, {"score": 0.7})

        runner.plot_ablation_results(["score"], output_dir=str(tmp_path))
        assert (tmp_path / "ablation_plots.png").exists()

    def test_empty_metrics_no_error(self, base_config, tmp_path):
        runner = AblationRunner(base_config, [], str(tmp_path))
        runner.plot_ablation_results([])  # should not raise


# ---------------------------------------------------------------------------
# Ablation group constants
# ---------------------------------------------------------------------------

class TestAblationGroupConstants:
    def test_names_are_unique_within_all_ablations(self):
        names = [a.get("name") for a in ALL_ABLATIONS]
        assert len(names) == len(set(names)), "Duplicate ablation names found"

    def test_all_ablations_have_name_key(self):
        for abl in ALL_ABLATIONS:
            assert "name" in abl, f"Ablation missing 'name' key: {abl}"

    def test_architecture_ablations_have_model_key(self):
        for abl in ARCHITECTURE_ABLATIONS:
            assert "model" in abl
            assert "architecture" in abl["model"]

    def test_loss_ablations_have_training_key(self):
        for abl in LOSS_ABLATIONS:
            assert "training" in abl
            assert "loss_name" in abl["training"]

    def test_masking_ablations_have_data_key(self):
        for abl in MASKING_ABLATIONS:
            assert "data" in abl
            assert "masking_base_rate" in abl["data"]

    def test_embedding_dim_ablations_have_model_key(self):
        for abl in EMBEDDING_DIM_ABLATIONS:
            assert "model" in abl
            assert "d_embed" in abl["model"]
            assert "d_model" in abl["model"]

    def test_all_ablations_count(self):
        expected = (
            len(ARCHITECTURE_ABLATIONS)
            + len(LOSS_ABLATIONS)
            + len(MASKING_ABLATIONS)
            + len(EMBEDDING_DIM_ABLATIONS)
        )
        assert len(ALL_ABLATIONS) == expected
