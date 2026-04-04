"""End-to-end smoke test for the training pipeline.

Exercises the full path: config → data load → distance matrix → dataset →
one training epoch → artifact write.  Uses tiny synthetic fixtures so the
test completes quickly.

Run with:  uv run pytest -m slow tests/test_smoke.py
Skip with: uv run pytest -m "not slow"
"""

from __future__ import annotations

import pytest
import polars as pl
import numpy as np


@pytest.mark.slow
def test_training_smoke(tmp_path):
    """Full pipeline on tiny synthetic data: must complete without error and
    return a metrics dict containing at least ``meta/n_params``."""
    from experiments.configs import (
        DataConfig,
        EvaluationConfig,
        ExperimentConfig,
        ModelConfig,
        TrainingConfig,
    )
    from experiments.run import run_experiment_returning_metrics

    n_persons = 10
    pids = [f"p{i}" for i in range(n_persons)]

    # ------------------------------------------------------------------
    # Synthetic activities — four distinct patterns so pairwise distances
    # are non-zero and the training signal is non-trivial.
    # Each pattern covers exactly 1440 minutes.
    # ------------------------------------------------------------------
    _patterns = {
        "A": [("home", 0, 480), ("work", 480, 960), ("home", 960, 1440)],
        "B": [("home", 0, 480), ("education", 480, 840), ("leisure", 840, 1440)],
        "C": [("home", 0, 600), ("shop", 600, 720), ("home", 720, 1440)],
        "D": [
            ("home", 0, 240),
            ("work", 240, 720),
            ("leisure", 720, 960),
            ("home", 960, 1440),
        ],
    }
    _pattern_keys = ["A", "B", "C", "D", "A", "B", "C", "D", "A", "B"]
    act_rows = []
    for pid, pat_key in zip(pids, _pattern_keys):
        for seq, (act, start, end) in enumerate(_patterns[pat_key]):
            act_rows.append(
                {
                    "pid": pid,
                    "seq": seq,
                    "act": act,
                    "zone": "z1",
                    "start": start,
                    "end": end,
                }
            )
    activities_df = pl.DataFrame(act_rows).cast(
        {
            "pid": pl.Utf8,
            "seq": pl.Int32,
            "act": pl.Utf8,
            "zone": pl.Utf8,
            "start": pl.Int32,
            "end": pl.Int32,
        }
    )
    activities_csv = tmp_path / "activities.csv"
    activities_df.write_csv(activities_csv)

    # ------------------------------------------------------------------
    # Synthetic attributes (one row per person)
    # ------------------------------------------------------------------
    rng = np.random.default_rng(0)
    attr_rows = []
    for i, pid in enumerate(pids):
        attr_rows.append(
            {
                "pid": pid,
                "hid": f"h{i}",
                "age": int(20 + i * 3),
                "hh_size": int(1 + i % 4),
                "hh_income": float(30000 + i * 5000),
                "sex": "M" if i % 2 == 0 else "F",
                "dwelling": "house",
                "ownership": "own",
                "vehicles": int(i % 3),
                "disability": None,
                "education": "secondary",
                "can_wfh": "yes" if i % 2 == 0 else "no",
                "occupation": "employed",
                "race": None,
                "has_licence": "yes",
                "relationship": "single",
                "employment": "full_time",
                "country": "AU",
                "source": "src_a" if i < 5 else "src_b",
                "year": 2019,
                "month": "march",
                "day": "tuesday",
                "hh_zone": f"zone_{i % 3}",
                "weight": 1.0,
                "access_egress_distance": float(rng.uniform(0.1, 5.0)),
                "max_temp_c": float(rng.uniform(10.0, 35.0)),
                "rain": "no",
                "avg_speed": float(rng.uniform(5.0, 30.0)),
            }
        )
    attributes_df = pl.DataFrame(attr_rows)
    attributes_csv = tmp_path / "attributes.csv"
    attributes_df.write_csv(attributes_csv)

    # ------------------------------------------------------------------
    # Minimal experiment config (no YAML, programmatic)
    # ------------------------------------------------------------------
    config = ExperimentConfig(
        name="smoke",
        seed=0,
        output_dir=str(tmp_path / "outputs"),
        data=DataConfig(
            data_path=str(activities_csv),
            attributes_path=str(attributes_csv),
            # weights as dict — _run_training must normalise to tuple
            distance_weights={
                "participation": 1 / 3,
                "sequence": 1 / 3,
                "timing": 1 / 3,
            },
            train_fraction=0.6,
            val_fraction=0.2,
            batch_size=4,
            num_workers=0,
            mode="pairwise",
            n_val_pairs=2,
            masking_base_rate=0.0,
        ),
        model=ModelConfig(
            architecture="addition",
            d_embed=8,
            d_model=16,
        ),
        training=TrainingConfig(
            loss_name="distance_regression",
            max_epochs=1,
            warmup_steps=0,
            log_every_n_steps=1,
        ),
        evaluation=EvaluationConfig(run_downstream=False),
    )

    # ------------------------------------------------------------------
    # Run and assert
    # ------------------------------------------------------------------
    metrics = run_experiment_returning_metrics(config)

    assert metrics, "metrics dict should not be empty"
    assert (
        "meta/n_params" in metrics
    ), f"expected meta/n_params, got keys: {list(metrics)}"
    assert metrics["meta/n_params"] > 0
