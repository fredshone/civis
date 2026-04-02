"""Ablation study framework for schedule embedding experiments.

Systematically runs all planned ablation groups (architecture, loss function,
masking augmentation, embedding dimension) with multiple random seeds, then
aggregates results into comparison tables and plots.

Public API
----------
AblationRunner
    Orchestrates multi-seed ablation sweeps, result persistence, aggregation,
    comparison table generation, and plotting.

_apply_overrides(base, overrides) -> ExperimentConfig
    Deep-merge a nested override dict into a copy of the base config.

ARCHITECTURE_ABLATIONS
    Six architecture variants: addition, FiLM, attention (1L/1H through 4L/8H).

LOSS_ABLATIONS
    Six loss variants: distance regression (MSE/Huber), soft-NN (fixed/learned tau),
    rank correlation, NT-Xent.

MASKING_ABLATIONS
    Six masking variants: none, low (5%), medium (15%), high (30%),
    grouped, curriculum.

EMBEDDING_DIM_ABLATIONS
    Three embedding dimension pairs: (32, 64), (64, 128), (128, 256).

ALL_ABLATIONS
    Concatenation of all four ablation groups.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.configs import (
    DataConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
)


# ---------------------------------------------------------------------------
# Ablation group definitions
# ---------------------------------------------------------------------------

ARCHITECTURE_ABLATIONS: list[dict[str, Any]] = [
    {
        "name": "arch_addition",
        "model": {"architecture": "addition"},
    },
    {
        "name": "arch_film",
        "model": {"architecture": "film"},
    },
    {
        "name": "arch_attention_1L_1H",
        "model": {"architecture": "attention", "n_layers": 1, "n_heads": 1},
    },
    {
        "name": "arch_attention_1L_4H",
        "model": {"architecture": "attention", "n_layers": 1, "n_heads": 4},
    },
    {
        "name": "arch_attention_2L_4H",
        "model": {"architecture": "attention", "n_layers": 2, "n_heads": 4},
    },
    {
        "name": "arch_attention_4L_8H",
        "model": {"architecture": "attention", "n_layers": 4, "n_heads": 8},
    },
]

LOSS_ABLATIONS: list[dict[str, Any]] = [
    {
        "name": "loss_regression_mse",
        "training": {
            "loss_name": "distance_regression",
            "loss_kwargs": {"use_huber": False, "normalize_emb_dist": True},
        },
    },
    {
        "name": "loss_regression_huber",
        "training": {
            "loss_name": "distance_regression",
            "loss_kwargs": {"use_huber": True, "normalize_emb_dist": True},
        },
    },
    {
        "name": "loss_soft_nn_fixed",
        "training": {
            "loss_name": "soft_nearest_neighbour",
            "loss_kwargs": {"learnable_tau": False},
        },
    },
    {
        "name": "loss_soft_nn_learned",
        "training": {
            "loss_name": "soft_nearest_neighbour",
            "loss_kwargs": {"learnable_tau": True},
        },
    },
    {
        "name": "loss_rank_correlation",
        "training": {
            "loss_name": "rank_correlation",
            "loss_kwargs": {},
        },
    },
    {
        "name": "loss_ntxent",
        "training": {
            "loss_name": "ntxent",
            "loss_kwargs": {},
        },
    },
]

MASKING_ABLATIONS: list[dict[str, Any]] = [
    {
        "name": "masking_none",
        "data": {"masking_base_rate": 0.0, "masking_strategy": "independent"},
    },
    {
        "name": "masking_low",
        "data": {"masking_base_rate": 0.05, "masking_strategy": "independent"},
    },
    {
        "name": "masking_medium",
        "data": {"masking_base_rate": 0.15, "masking_strategy": "independent"},
    },
    {
        "name": "masking_high",
        "data": {"masking_base_rate": 0.30, "masking_strategy": "independent"},
    },
    {
        "name": "masking_grouped",
        "data": {"masking_base_rate": 0.15, "masking_strategy": "grouped"},
    },
    {
        "name": "masking_curriculum",
        "data": {"masking_base_rate": 0.15, "masking_strategy": "curriculum"},
    },
]

EMBEDDING_DIM_ABLATIONS: list[dict[str, Any]] = [
    {
        "name": "dims_32_64",
        "model": {"d_embed": 32, "d_model": 64},
    },
    {
        "name": "dims_64_128",
        "model": {"d_embed": 64, "d_model": 128},
    },
    {
        "name": "dims_128_256",
        "model": {"d_embed": 128, "d_model": 256},
    },
]

ALL_ABLATIONS: list[dict[str, Any]] = (
    ARCHITECTURE_ABLATIONS
    + LOSS_ABLATIONS
    + MASKING_ABLATIONS
    + EMBEDDING_DIM_ABLATIONS
)


# ---------------------------------------------------------------------------
# Config override helper
# ---------------------------------------------------------------------------

def _apply_overrides(
    base: ExperimentConfig,
    overrides: dict[str, Any],
) -> ExperimentConfig:
    """Return a new ExperimentConfig with override values shallow-merged.

    Top-level scalar fields (``"name"``, ``"seed"``, ``"output_dir"``) are
    replaced directly.  Sub-config keys (``"data"``, ``"model"``,
    ``"training"``, ``"evaluation"``) are shallow-merged into the
    corresponding base sub-config, replacing individual fields while
    preserving unmentioned ones.

    The original ``base`` is never mutated.

    Parameters
    ----------
    base:
        Template configuration to copy.
    overrides:
        Two-level dict of overrides.  Example::

            {
                "name": "attention_1L_1H",
                "model": {"architecture": "attention", "n_layers": 1, "n_heads": 1},
            }

    Returns
    -------
    ExperimentConfig
        New configuration with overrides applied.
    """
    base_dict = dataclasses.asdict(base)

    new_data = dict(base_dict["data"])
    new_model = dict(base_dict["model"])
    new_training = dict(base_dict["training"])
    new_evaluation = dict(base_dict["evaluation"])

    if "data" in overrides:
        new_data.update(overrides["data"])
    if "model" in overrides:
        new_model.update(overrides["model"])
    if "training" in overrides:
        new_training.update(overrides["training"])
    if "evaluation" in overrides:
        new_evaluation.update(overrides["evaluation"])

    return ExperimentConfig(
        name=overrides.get("name", base.name),
        seed=overrides.get("seed", base.seed),
        output_dir=overrides.get("output_dir", base.output_dir),
        data=DataConfig(**{
            k: v for k, v in new_data.items()
            if k in DataConfig.__dataclass_fields__
        }),
        model=ModelConfig(**{
            k: v for k, v in new_model.items()
            if k in ModelConfig.__dataclass_fields__
        }),
        training=TrainingConfig(**{
            k: v for k, v in new_training.items()
            if k in TrainingConfig.__dataclass_fields__
        }),
        evaluation=EvaluationConfig(**{
            k: v for k, v in new_evaluation.items()
            if k in EvaluationConfig.__dataclass_fields__
        }),
    )


# ---------------------------------------------------------------------------
# AblationRunner
# ---------------------------------------------------------------------------

class AblationRunner:
    """Systematically run ablation studies and aggregate results.

    Each ablation config is run with ``n_seeds`` random seeds.  Results are
    written to disk as JSON immediately after each run, so interrupted sweeps
    can be resumed without re-running completed experiments.

    Parameters
    ----------
    base_config:
        Template :class:`~experiments.configs.ExperimentConfig` that all
        ablations derive from.  Per-ablation per-seed runs override the
        ``name``, ``seed``, and ``output_dir`` fields automatically.
    ablation_configs:
        List of override dicts.  Each **must** have a ``"name"`` key and may
        have ``"model"``, ``"training"``, ``"data"``, or ``"evaluation"``
        sub-dicts that are shallow-merged into the corresponding base
        sub-config.
    output_base_dir:
        Root directory for all outputs.  Layout::

            <output_base_dir>/
              <ablation_name>/
                seed_0/results.json
                seed_1/results.json
                ...
    """

    def __init__(
        self,
        base_config: ExperimentConfig,
        ablation_configs: list[dict[str, Any]],
        output_base_dir: str = "outputs/ablations",
    ) -> None:
        self.base_config = base_config
        self.ablation_configs = ablation_configs
        self.output_base_dir = Path(output_base_dir)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run_all(self, n_seeds: int = 3) -> None:
        """Run every ablation × seed combination.

        Skips any ``(ablation, seed)`` pair whose ``results.json`` already
        exists, allowing interrupted sweeps to be resumed.

        Parameters
        ----------
        n_seeds:
            Number of random seeds to run per ablation.  Seeds are derived
            as ``base_config.seed + seed_offset`` for ``seed_offset`` in
            ``range(n_seeds)``.
        """
        from experiments.run import run_experiment_returning_metrics

        n_total = len(self.ablation_configs) * n_seeds
        done = 0

        for abl_idx, ablation in enumerate(self.ablation_configs):
            ablation_name = ablation.get("name", f"ablation_{abl_idx}")

            for seed_offset in range(n_seeds):
                done += 1
                run_dir = self.output_base_dir / ablation_name / f"seed_{seed_offset}"
                result_path = run_dir / "results.json"

                if result_path.exists():
                    print(
                        f"  [{done}/{n_total}] Skipping {ablation_name}/"
                        f"seed_{seed_offset} (already done)"
                    )
                    continue

                # Build per-seed config
                cfg = _apply_overrides(self.base_config, ablation)
                cfg = dataclasses.replace(
                    cfg,
                    name=f"{ablation_name}_seed{seed_offset}",
                    seed=self.base_config.seed + seed_offset,
                    output_dir=str(self.output_base_dir),
                )

                print(
                    f"\n[AblationRunner {done}/{n_total}] "
                    f"{ablation_name} seed={seed_offset}"
                )

                t0 = time.monotonic()
                try:
                    metrics = run_experiment_returning_metrics(cfg)
                    metrics["meta/train_time_s"] = time.monotonic() - t0
                except Exception as exc:
                    print(f"  ERROR during {ablation_name}/seed_{seed_offset}: {exc}")
                    metrics = {"meta/error": str(exc)}

                run_dir.mkdir(parents=True, exist_ok=True)
                with open(result_path, "w") as fh:
                    json.dump(metrics, fh, indent=2)
                print(f"  Saved → {result_path}")

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def aggregate_results(
        self,
    ) -> dict[str, dict[str, tuple[float, float]]]:
        """Load per-seed JSON results and compute mean ± std per ablation.

        Only metrics that are numeric and present in every seed result for
        an ablation are included (metrics missing in any seed are skipped).

        Returns
        -------
        dict mapping ablation name → {metric_key: (mean, std)}.
        An ablation without any completed seeds is absent from the dict.
        """
        aggregated: dict[str, dict[str, tuple[float, float]]] = {}

        for abl_idx, ablation in enumerate(self.ablation_configs):
            ablation_name = ablation.get("name", f"ablation_{abl_idx}")
            ablation_dir = self.output_base_dir / ablation_name

            if not ablation_dir.exists():
                continue

            seed_results: list[dict[str, float]] = []
            for result_path in sorted(ablation_dir.glob("seed_*/results.json")):
                with open(result_path) as fh:
                    raw = json.load(fh)
                numeric = {k: v for k, v in raw.items() if isinstance(v, (int, float))}
                if numeric:
                    seed_results.append(numeric)

            if not seed_results:
                continue

            # Keep only keys present in every seed
            common_keys = set(seed_results[0])
            for r in seed_results[1:]:
                common_keys &= set(r)

            agg: dict[str, tuple[float, float]] = {}
            for key in sorted(common_keys):
                vals = np.array([r[key] for r in seed_results], dtype=float)
                agg[key] = (float(np.nanmean(vals)), float(np.nanstd(vals)))

            aggregated[ablation_name] = agg

        return aggregated

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------

    def comparison_table(self, metrics: list[str]) -> pd.DataFrame:
        """Build a paper-ready comparison table.

        Parameters
        ----------
        metrics:
            Metric keys to include as columns, in the order they should appear.

        Returns
        -------
        pd.DataFrame
            Index: ablation names (in the order of ``self.ablation_configs``).
            Columns: one per requested metric.
            Cells: ``"mean ± std"`` strings (``"—"`` when data is absent).
        """
        aggregated = self.aggregate_results()
        rows = []
        index = []

        for abl_idx, ablation in enumerate(self.ablation_configs):
            name = ablation.get("name", f"ablation_{abl_idx}")
            agg = aggregated.get(name, {})
            row: dict[str, str] = {}
            for m in metrics:
                if m in agg:
                    mean, std = agg[m]
                    row[m] = f"{mean:.4f} ± {std:.4f}"
                else:
                    row[m] = "—"
            rows.append(row)
            index.append(name)

        return pd.DataFrame(rows, index=index, columns=metrics)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_ablation_results(
        self,
        metrics: list[str],
        output_dir: str | None = None,
    ) -> None:
        """Produce a multi-panel figure showing metric variation across ablations.

        One panel per metric.  Ablation variants are on the x-axis; the mean
        ± std is shown as a bar chart with error bars.  The figure is saved
        to disk as ``ablation_plots.png``.

        Parameters
        ----------
        metrics:
            Metric keys to plot.
        output_dir:
            Directory to save the figure.  Defaults to
            ``self.output_base_dir``.
        """
        aggregated = self.aggregate_results()

        out_path = Path(output_dir) if output_dir else self.output_base_dir
        out_path.mkdir(parents=True, exist_ok=True)

        n_metrics = len(metrics)
        if n_metrics == 0:
            return

        ncols = min(n_metrics, 3)
        nrows = (n_metrics + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
        )

        for idx, metric in enumerate(metrics):
            ax = axes[idx // ncols][idx % ncols]
            names: list[str] = []
            means: list[float] = []
            stds: list[float] = []

            for abl_idx, ablation in enumerate(self.ablation_configs):
                name = ablation.get("name", f"ablation_{abl_idx}")
                agg = aggregated.get(name, {})
                if metric in agg:
                    mean, std = agg[metric]
                    names.append(name)
                    means.append(mean)
                    stds.append(std)

            if not names:
                ax.set_visible(False)
                continue

            x = np.arange(len(names))
            ax.bar(x, means, yerr=stds, capsize=4, alpha=0.75, color="steelblue")
            ax.set_xticks(x)
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
            ax.set_title(metric, fontsize=9)
            ax.set_ylabel("value")
            ax.grid(axis="y", alpha=0.3)

        # Hide unused subplot panels
        for idx in range(n_metrics, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.tight_layout()
        save_path = out_path / "ablation_plots.png"
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved ablation plot → {save_path}")
