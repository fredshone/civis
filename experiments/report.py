"""Reporting and visualisation for schedule embedding experiments.

Generates UMAP projections of learned embeddings, per-group ablation comparison
tables, a narrative ``FINDINGS.md``, and the master ``RESULTS.md`` document
that assembles all outputs after a complete experiment run.

All functions that read ablation results operate purely on the JSON files
written by :class:`~experiments.ablations.AblationRunner` — no experiments are
re-run.  The UMAP functions require a trained model and load one from an
experiment output directory (see :func:`generate_umap_from_dir`).

Public API
----------
seed_everything(seed)
    Set all random seeds for reproducibility.

plot_umap_projections(embedder, test_attributes, encoder, D_test, output_dir, seed)
    Generate and save three UMAP scatter plots.

generate_umap_from_dir(model_dir, output_dir, seed)
    Load a trained model from disk and generate UMAP projections.

generate_results_tables(ablation_results_dir, output_dir, base_config)
    Write per-group ablation comparison tables as markdown files.

generate_findings(ablation_results_dir, output_dir, base_config)
    Write FINDINGS.md summarising key results.

generate_report(results_dir, output_dir, base_config)
    Assemble master RESULTS.md from tables, figures, and findings.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.ablations import (
    ARCHITECTURE_ABLATIONS,
    EMBEDDING_DIM_ABLATIONS,
    LOSS_ABLATIONS,
    MASKING_ABLATIONS,
    AblationRunner,
)
from experiments.configs import ExperimentConfig


# ---------------------------------------------------------------------------
# Metric sets used in each ablation table
# ---------------------------------------------------------------------------

_REPORT_METRICS: list[str] = [
    "intrinsic/rank_correlation",
    "discrete/work_participation_linear_roc_auc",
    "continuous/work_duration_linear_r2",
    "continuous/trip_count_linear_r2",
]

_DIM_METRICS: list[str] = ["meta/n_params"] + _REPORT_METRICS

_ABLATION_GROUPS: dict[str, list[dict]] = {
    "architecture": ARCHITECTURE_ABLATIONS,
    "loss": LOSS_ABLATIONS,
    "masking": MASKING_ABLATIONS,
    "embedding_dims": EMBEDDING_DIM_ABLATIONS,
}

_GROUP_METRICS: dict[str, list[str]] = {
    "architecture": _REPORT_METRICS,
    "loss": _REPORT_METRICS,
    "masking": _REPORT_METRICS,
    "embedding_dims": _DIM_METRICS,
}


# ---------------------------------------------------------------------------
# Seed management
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    """Set torch, numpy, and random module seeds.

    Parameters
    ----------
    seed:
        Integer seed applied to ``random``, ``numpy.random``, ``torch``,
        and ``torch.cuda``.
    """
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# UMAP visualisation
# ---------------------------------------------------------------------------

def plot_umap_projections(
    embedder: "torch.nn.Module",
    test_attributes: "pl.DataFrame",
    encoder: "AttributeEncoder",
    D_test: np.ndarray,
    output_dir: str | Path,
    seed: int = 42,
) -> list[Path]:
    """Generate UMAP scatter plots of test-set embeddings.

    Produces three figures saved under ``{output_dir}/figures/``:

    - ``umap_source.png``: coloured by data source
    - ``umap_employment.png``: coloured by employment status
    - ``umap_schedule_cluster.png``: coloured by k-means clusters on D_test

    Parameters
    ----------
    embedder:
        Trained :class:`~models.base.BaseAttributeEmbedder` in eval mode.
    test_attributes:
        Polars DataFrame of test-set person attributes.
    encoder:
        Fitted :class:`~datasets.encoding.AttributeEncoder`.
    D_test:
        Pairwise schedule distance matrix for the test set, shape ``(N, N)``.
    output_dir:
        Directory under which a ``figures/`` subdirectory is created.
    seed:
        Random seed for UMAP and k-means.

    Returns
    -------
    list[Path]
        Paths of saved figure files.
    """
    import torch
    from sklearn.cluster import KMeans

    try:
        import umap as _umap
    except ImportError as exc:
        raise ImportError(
            "umap-learn is required for UMAP visualisation: uv add umap-learn"
        ) from exc

    figures_dir = Path(output_dir) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)

    # --- Compute embeddings in batches ---
    embedder.eval()
    device = next(embedder.parameters()).device
    attrs = encoder.transform(test_attributes)
    attrs = {k: v.to(device) for k, v in attrs.items()}

    batch_size = 512
    n_samples = next(iter(attrs.values())).shape[0]
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            chunk = {k: v[start : start + batch_size] for k, v in attrs.items()}
            chunks.append(embedder(chunk).cpu().numpy())
    embeddings = np.vstack(chunks)  # (N, d_model)

    # --- Fit UMAP ---
    reducer = _umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        random_state=seed,
    )
    coords = reducer.fit_transform(embeddings)  # (N, 2)

    saved: list[Path] = []

    # --- Plot 1: colour by source ---
    if "source" in test_attributes.columns:
        labels = [str(v) if v is not None else "unknown"
                  for v in test_attributes["source"].to_list()]
        path = figures_dir / "umap_source.png"
        _save_scatter(coords, labels, path, title="UMAP — coloured by source")
        saved.append(path)

    # --- Plot 2: colour by employment ---
    if "employment" in test_attributes.columns:
        labels = [str(v) if v is not None else "unknown"
                  for v in test_attributes["employment"].to_list()]
        path = figures_dir / "umap_employment.png"
        _save_scatter(
            coords, labels, path,
            title="UMAP — coloured by employment status",
        )
        saved.append(path)

    # --- Plot 3: colour by schedule cluster ---
    n_clusters = 8
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    cluster_labels = [str(c) for c in km.fit_predict(D_test).tolist()]
    path = figures_dir / "umap_schedule_cluster.png"
    _save_scatter(
        coords, cluster_labels, path,
        title=f"UMAP — coloured by schedule cluster (k={n_clusters})",
    )
    saved.append(path)

    return saved


def _save_scatter(
    coords: np.ndarray,
    labels: list[str],
    path: Path,
    title: str,
    alpha: float = 0.4,
    s: float = 5.0,
) -> None:
    """Save a 2D scatter plot with discrete colour coding."""
    unique = sorted(set(labels))
    palette = plt.get_cmap("tab20", max(len(unique), 1))
    label_to_idx = {u: i for i, u in enumerate(unique)}

    fig, ax = plt.subplots(figsize=(7, 6))
    for u in unique:
        mask = np.array([l == u for l in labels])
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[palette(label_to_idx[u])],
            label=u, s=s, alpha=alpha,
        )

    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(
        markerscale=3, loc="best",
        fontsize=7, ncol=max(1, len(unique) // 10),
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {path}")


def generate_umap_from_dir(
    model_dir: str | Path,
    output_dir: str | Path,
    seed: int | None = None,
) -> list[Path]:
    """Load a trained model from an experiment directory and generate UMAP plots.

    Reads the following files written by
    :func:`~experiments.run._run_training`:

    - ``config.json``: experiment configuration dict
    - ``encoder.pkl``: fitted :class:`~datasets.encoding.AttributeEncoder`
    - ``model.pt``: ``model.state_dict()``
    - ``distance_matrix.npy``: precomputed N×N pairwise distance matrix

    Parameters
    ----------
    model_dir:
        Experiment output directory (e.g. ``outputs/attention_2layer``).
    output_dir:
        Where to write figures.
    seed:
        Override random seed.  Defaults to the value in ``config.json``.

    Returns
    -------
    list[Path]
        Paths of saved figure files.
    """
    import dataclasses

    import numpy as np
    import torch
    from datasets.encoding import AttributeEncoder
    from distances.data import load_activities, load_attributes
    from models.base import AttributeEmbedderConfig
    from models.registry import build_model

    model_dir = Path(model_dir)

    with open(model_dir / "config.json") as fh:
        cfg = json.load(fh)

    exp_seed = seed if seed is not None else cfg.get("seed", 42)
    seed_everything(exp_seed)

    encoder = AttributeEncoder.load(model_dir / "encoder.pkl")

    data_cfg = cfg["data"]
    activities_df = load_activities(data_cfg["data_path"])
    attributes_df = load_attributes(data_cfg["attributes_path"])
    N = len(attributes_df)

    D = np.load(model_dir / "distance_matrix.npy")

    # Reproduce the same train/val/test split used during training
    rng = np.random.default_rng(exp_seed)
    indices = rng.permutation(N)
    n_train = int(N * data_cfg["train_fraction"])
    n_val = int(N * data_cfg["val_fraction"])
    test_idx = indices[n_train + n_val :]

    test_attributes_df = attributes_df[test_idx.tolist()]
    D_test = D[np.ix_(test_idx, test_idx)]

    # Rebuild model from saved config
    model_cfg = cfg["model"]
    embedder_config = AttributeEmbedderConfig.from_encoder(
        encoder,
        d_embed=model_cfg["d_embed"],
        d_model=model_cfg["d_model"],
        dropout=model_cfg.get("dropout", 0.1),
        n_heads=model_cfg.get("n_heads", 4),
        n_layers=model_cfg.get("n_layers", 2),
        use_cls_token=model_cfg.get("use_cls_token", True),
        pooling=model_cfg.get("pooling", "cls"),
        context_attributes=model_cfg.get("context_attributes") or [],
        attribute_groups=model_cfg.get("attribute_groups"),
    )
    model = build_model({
        "architecture": model_cfg["architecture"],
        **dataclasses.asdict(embedder_config),
    })
    model.load_state_dict(
        torch.load(model_dir / "model.pt", map_location="cpu", weights_only=True)
    )
    model.eval()

    return plot_umap_projections(
        embedder=model,
        test_attributes=test_attributes_df,
        encoder=encoder,
        D_test=D_test,
        output_dir=output_dir,
        seed=exp_seed,
    )


# ---------------------------------------------------------------------------
# Table and findings generation (JSON-only, no model needed)
# ---------------------------------------------------------------------------

def generate_results_tables(
    ablation_results_dir: str | Path,
    output_dir: str | Path,
    base_config: ExperimentConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Generate per-group ablation comparison tables as markdown files.

    Reads existing result JSON files; does not re-run any experiments.

    Parameters
    ----------
    ablation_results_dir:
        Root directory written by :class:`~experiments.ablations.AblationRunner`.
    output_dir:
        Where to write ``tables/table_{group}.md`` files.
    base_config:
        Placeholder config passed to AblationRunner.  Can be ``None``.

    Returns
    -------
    dict
        Mapping group name → :class:`pandas.DataFrame`.
    """
    tables_dir = Path(output_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    if base_config is None:
        base_config = _dummy_config()

    tables: dict[str, pd.DataFrame] = {}
    for group_name, ablations in _ABLATION_GROUPS.items():
        runner = AblationRunner(base_config, ablations, str(ablation_results_dir))
        metrics = _GROUP_METRICS[group_name]
        df = runner.comparison_table(metrics)
        tables[group_name] = df

        md_path = tables_dir / f"table_{group_name}.md"
        with open(md_path, "w") as fh:
            fh.write(f"## {group_name.replace('_', ' ').title()} Ablation\n\n")
            fh.write(df.to_markdown() or "*(no data)*")
            fh.write("\n")
        print(f"Saved table → {md_path}")

    return tables


def generate_findings(
    ablation_results_dir: str | Path,
    output_dir: str | Path,
    base_config: ExperimentConfig | None = None,
) -> Path:
    """Write FINDINGS.md summarising key results from ablation experiments.

    Parameters
    ----------
    ablation_results_dir:
        Root directory of :class:`~experiments.ablations.AblationRunner` outputs.
    output_dir:
        Where to write ``FINDINGS.md``.
    base_config:
        Placeholder config passed to AblationRunner.  Can be ``None``.

    Returns
    -------
    Path
        Path to the written ``FINDINGS.md``.
    """
    if base_config is None:
        base_config = _dummy_config()

    # Aggregate all results across groups
    all_agg: dict[str, dict[str, tuple[float, float]]] = {}
    for ablations in _ABLATION_GROUPS.values():
        runner = AblationRunner(base_config, ablations, str(ablation_results_dir))
        all_agg.update(runner.aggregate_results())

    primary = "intrinsic/rank_correlation"

    arch_names = [a["name"] for a in ARCHITECTURE_ABLATIONS]
    best_arch = _best_by(all_agg, arch_names, primary)
    best_arch_rc = all_agg.get(best_arch or "", {}).get(primary, (float("nan"),))[0]

    loss_names = [a["name"] for a in LOSS_ABLATIONS]
    best_loss = _best_by(all_agg, loss_names, primary)
    best_loss_rc = all_agg.get(best_loss or "", {}).get(primary, (float("nan"),))[0]

    mask_none_rc = all_agg.get("masking_none", {}).get(primary, (float("nan"), 0))[0]
    mask_names = [a["name"] for a in MASKING_ABLATIONS]
    best_mask = _best_by(all_agg, mask_names, primary)
    best_mask_rc = all_agg.get(best_mask or "", {}).get(primary, (float("nan"), 0))[0]
    masking_delta = best_mask_rc - mask_none_rc

    downstream_tasks = [
        ("Work participation AUC", "discrete/work_participation_linear_roc_auc"),
        ("Work duration R²",       "continuous/work_duration_linear_r2"),
        ("Trip count R²",          "continuous/trip_count_linear_r2"),
    ]
    best_task = _most_sensitive_task(all_agg, arch_names, downstream_tasks)

    def _fmt(v: float) -> str:
        return f"{v:.4f}" if not np.isnan(v) else "N/A"

    lines = [
        "# Findings\n",
        "## 1. Best Architecture",
        "",
        f"Architecture with highest intrinsic rank correlation: **{best_arch or 'unknown'}**  ",
        f"Rank correlation = {_fmt(best_arch_rc)}",
        "",
        "## 2. Most Effective Loss Function",
        "",
        f"Loss with highest intrinsic rank correlation: **{best_loss or 'unknown'}**  ",
        f"Rank correlation = {_fmt(best_loss_rc)}",
        "",
        "## 3. Impact of Masking Augmentation",
        "",
        f"No-masking rank correlation: {_fmt(mask_none_rc)}  ",
        f"Best masking strategy (`{best_mask}`): {_fmt(best_mask_rc)}  ",
        f"Delta from masking augmentation: **{masking_delta:+.4f}**",
        "",
        "## 4. Downstream Task Sensitivity",
        "",
        f"Task most improved by better embeddings (largest range across architectures):"
        f" **{best_task or 'unknown'}**",
        "",
        "## 5. Attention Weights",
        "",
        "See `figures/umap_*.png` and the per-experiment `geometry/` report for "
        "attention heatmaps and embedding structure.",
        "",
        "## 6. Conditional Collapse",
        "",
        "Refer to the generative evaluation section of RESULTS.md for collapse "
        "profiles from the best and worst models.",
        "",
        "## Reproducibility",
        "",
        "All experiments are seeded deterministically via `ExperimentConfig.seed`.",
        "Multi-seed ablations use seeds `base_seed + offset` for `offset` in `range(n_seeds)`.",
        "PRNG libraries seeded: `torch`, `torch.cuda`, `numpy.random`, `random`.",
    ]

    output_path = Path(output_dir) / "FINDINGS.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Saved → {output_path}")
    return output_path


def generate_report(
    results_dir: str | Path,
    output_dir: str | Path,
    base_config: ExperimentConfig | None = None,
) -> Path:
    """Assemble the master RESULTS.md from tables, figures, and findings.

    Expects ablation runner outputs under ``{results_dir}/ablations/``.

    Parameters
    ----------
    results_dir:
        Root outputs directory (parent of ``ablations/`` subdirectory).
    output_dir:
        Where to write ``RESULTS.md``, ``tables/``, ``figures/``, and
        ``FINDINGS.md``.
    base_config:
        Placeholder config passed to AblationRunner.  Can be ``None``.

    Returns
    -------
    Path
        Path to the written ``RESULTS.md``.
    """
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ablation_dir = results_dir / "ablations"

    if base_config is None:
        base_config = _dummy_config()

    tables = generate_results_tables(ablation_dir, output_dir, base_config)
    findings_path = generate_findings(ablation_dir, output_dir, base_config)

    # Per-group ablation plots
    for group_name, ablations in _ABLATION_GROUPS.items():
        runner = AblationRunner(base_config, ablations, str(ablation_dir))
        runner.plot_ablation_results(
            _GROUP_METRICS[group_name],
            output_dir=str(output_dir / "figures" / group_name),
        )

    # --- Assemble RESULTS.md ---
    sections: list[str] = [
        "# Results\n",
        "_Auto-generated by `experiments/report.py`. "
        "Re-run `scripts/reproduce.sh` to refresh._\n",
        "",
        "## Architecture Comparison\n",
        tables.get("architecture", pd.DataFrame()).to_markdown() or "*(no data)*",
        "",
        "## Loss Function Comparison\n",
        tables.get("loss", pd.DataFrame()).to_markdown() or "*(no data)*",
        "",
        "## Masking Augmentation Comparison\n",
        tables.get("masking", pd.DataFrame()).to_markdown() or "*(no data)*",
        "",
        "## Embedding Dimension Comparison\n",
        tables.get("embedding_dims", pd.DataFrame()).to_markdown() or "*(no data)*",
        "",
        "## Embedding Visualisations\n",
        "UMAP projections of the best model's test-set embeddings:\n",
    ]

    umap_figs = [
        ("umap_source.png",          "UMAP coloured by data source"),
        ("umap_employment.png",      "UMAP coloured by employment status"),
        ("umap_schedule_cluster.png","UMAP coloured by schedule cluster"),
    ]
    for fname, caption in umap_figs:
        rel = Path("figures") / fname
        sections += [f"### {caption}\n", f"![{caption}]({rel})\n", ""]

    sections += [
        "## Ablation Plots\n",
        "![Architecture ablation](figures/architecture/ablation_plots.png)\n",
        "",
        "## Key Findings\n",
    ]
    if findings_path.exists():
        sections.append(findings_path.read_text())

    results_md = output_dir / "RESULTS.md"
    with open(results_md, "w") as fh:
        fh.write("\n".join(sections))
    print(f"Saved → {results_md}")
    return results_md


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_config() -> ExperimentConfig:
    """Return a minimal placeholder ExperimentConfig for AblationRunner."""
    from experiments.configs import (
        DataConfig,
        EvaluationConfig,
        ModelConfig,
        TrainingConfig,
    )
    return ExperimentConfig(
        name="dummy",
        seed=42,
        output_dir="outputs",
        data=DataConfig(),
        model=ModelConfig(),
        training=TrainingConfig(),
        evaluation=EvaluationConfig(),
    )


def _best_by(
    agg: dict[str, dict[str, tuple[float, float]]],
    names: list[str],
    metric: str,
) -> str | None:
    """Return the ablation name with the highest mean value for *metric*."""
    best_name: str | None = None
    best_val = float("-inf")
    for name in names:
        val = agg.get(name, {}).get(metric, (float("nan"), 0.0))[0]
        if not np.isnan(val) and val > best_val:
            best_val = val
            best_name = name
    return best_name


def _most_sensitive_task(
    agg: dict[str, dict[str, tuple[float, float]]],
    arch_names: list[str],
    tasks: list[tuple[str, str]],
) -> str | None:
    """Return the task name with the largest metric range across architectures."""
    best_task: str | None = None
    best_range = 0.0
    for task_name, key in tasks:
        vals = [
            agg.get(n, {}).get(key, (float("nan"), 0.0))[0]
            for n in arch_names
        ]
        vals = [v for v in vals if not np.isnan(v)]
        if len(vals) >= 2:
            r = max(vals) - min(vals)
            if r > best_range:
                best_range = r
                best_task = task_name
    return best_task


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate RESULTS.md and supporting figures from experiment outputs.\n\n"
            "Pass --umap-model-dir to also generate UMAP projections from a "
            "trained model checkpoint."
        )
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Root outputs directory (must contain an 'ablations/' subdirectory).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Where to write RESULTS.md, figures/, tables/, and FINDINGS.md.",
    )
    parser.add_argument(
        "--umap-model-dir",
        default=None,
        help=(
            "Experiment output directory containing config.json, encoder.pkl, "
            "model.pt, and distance_matrix.npy.  If given, UMAP plots are generated."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed (default: read from config.json).",
    )
    args = parser.parse_args()

    generate_report(args.results_dir, args.output_dir)

    if args.umap_model_dir:
        generate_umap_from_dir(args.umap_model_dir, args.output_dir, seed=args.seed)

    print(f"\nReport complete. See {args.output_dir}/RESULTS.md")
