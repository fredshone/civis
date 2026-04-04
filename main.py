"""CLI entry point for civis.

Usage
-----
::

    civis run experiments/configs/baseline_addition.yaml
    civis validate data/activities.parquet data/attributes.parquet
    civis ablate experiments/configs/baseline_addition.yaml
"""
from __future__ import annotations

import click


@click.group()
def cli() -> None:
    """Attribute embedding learning for human activity schedules."""


@cli.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--data-dir", default=None, help="Override data directory.")
def run(config: str, data_dir: str | None) -> None:
    """Run a training experiment from a YAML config."""
    from experiments.configs import load_config
    from experiments.run import _run_training, run_experiment
    if data_dir:
        import os
        cfg = load_config(config)
        cfg.data.data_path = os.path.join(data_dir, "activities.parquet")
        cfg.data.attributes_path = os.path.join(data_dir, "attributes.parquet")
        _run_training(cfg)
    else:
        run_experiment(config)


@cli.command()
@click.argument("activities", type=click.Path(exists=True))
@click.argument("attributes", type=click.Path(exists=True))
def validate(activities: str, attributes: str) -> None:
    """Load data files and print summary statistics."""
    from distances.data import load_activities, load_attributes, print_summary
    acts = load_activities(activities)
    attrs = load_attributes(attributes)
    print_summary(acts, attrs)


@cli.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--data-dir", default=None, help="Override data directory.")
@click.option("--output-dir", default="outputs/ablations", show_default=True,
              help="Root directory for ablation outputs.")
@click.option("--seeds", default=3, show_default=True,
              help="Number of random seeds per ablation.")
def ablate(config: str, data_dir: str | None, output_dir: str, seeds: int) -> None:
    """Run ablation study from a YAML config."""
    from experiments.ablations import ALL_ABLATIONS, AblationRunner
    from experiments.configs import load_config
    import os
    cfg = load_config(config)
    if data_dir:
        cfg.data.data_path = os.path.join(data_dir, "activities.parquet")
        cfg.data.attributes_path = os.path.join(data_dir, "attributes.parquet")
    runner = AblationRunner(
        base_config=cfg,
        ablation_configs=ALL_ABLATIONS,
        output_base_dir=output_dir,
    )
    runner.run_all(n_seeds=seeds)


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
