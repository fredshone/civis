#!/usr/bin/env bash
# reproduce.sh — end-to-end pipeline for civis attribute embedding experiments.
#
# Usage
# -----
#   DATA_DIR=/path/to/data bash scripts/reproduce.sh
#
# Environment variables
# ---------------------
# DATA_DIR          Directory containing activities.parquet and attributes.parquet.
#                   Default: data/
# OUTPUT_DIR        Root directory for all training outputs.
#                   Default: outputs
# RESULTS_DIR       Directory for the final report.
#                   Default: results
# ABLATION_SEEDS    Number of random seeds per ablation.
#                   Default: 3
# BASE_CONFIG       YAML config used as the template for ablation sweeps.
#                   Default: experiments/configs/attention_2layer.yaml
#
# The script is idempotent: each stage is skipped if its outputs already exist,
# so interrupted runs can be resumed.

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
RESULTS_DIR="${RESULTS_DIR:-results}"
ABLATION_SEEDS="${ABLATION_SEEDS:-3}"
BASE_CONFIG="${BASE_CONFIG:-experiments/configs/attention_2layer.yaml}"

ACTIVITIES_PATH="${DATA_DIR}/activities.parquet"
ATTRIBUTES_PATH="${DATA_DIR}/attributes.parquet"
DIST_CACHE="${OUTPUT_DIR}/distance_matrix.npy"

echo "========================================================"
echo " civis — Attribute Embedding Reproducibility Pipeline"
echo "========================================================"
echo "  DATA_DIR       = ${DATA_DIR}"
echo "  OUTPUT_DIR     = ${OUTPUT_DIR}"
echo "  RESULTS_DIR    = ${RESULTS_DIR}"
echo "  ABLATION_SEEDS = ${ABLATION_SEEDS}"
echo "  BASE_CONFIG    = ${BASE_CONFIG}"
echo ""

mkdir -p "${OUTPUT_DIR}" "${RESULTS_DIR}"

# ---------------------------------------------------------------------------
# Step 1 — Pre-compute composite distance matrix (cached)
# ---------------------------------------------------------------------------

echo "------------------------------------------------------------"
echo "Step 1: Schedule distance matrix"
echo "------------------------------------------------------------"

if [ -f "${DIST_CACHE}" ]; then
    echo "  Cached distance matrix found at ${DIST_CACHE} — skipping."
else
    echo "  Computing pairwise composite distance matrix …"
    uv run python - <<EOF
import numpy as np
from distances.composite import pairwise_composite_distance
from distances.data import load_activities
import pathlib

out = pathlib.Path("${DIST_CACHE}")
out.parent.mkdir(parents=True, exist_ok=True)
acts = load_activities("${ACTIVITIES_PATH}")
pids, D = pairwise_composite_distance(acts)
np.save(out, D)
print(f"  Saved distance matrix {D.shape} → {out}")
EOF
fi

# ---------------------------------------------------------------------------
# Step 2 — Baseline experiments (one per YAML config)
# ---------------------------------------------------------------------------

echo ""
echo "------------------------------------------------------------"
echo "Step 2: Baseline experiments"
echo "------------------------------------------------------------"

BASELINE_CONFIGS=(
    "experiments/configs/baseline_addition.yaml"
    "experiments/configs/attention_1layer.yaml"
    "experiments/configs/attention_2layer.yaml"
    "experiments/configs/attention_4layer.yaml"
    "experiments/configs/film.yaml"
)

for cfg in "${BASELINE_CONFIGS[@]}"; do
    # Extract experiment name from the yaml (first `name:` line)
    exp_name=$(grep -m1 '^name:' "${cfg}" | awk '{print $2}')
    marker="${OUTPUT_DIR}/${exp_name}/config.json"

    if [ -f "${marker}" ]; then
        echo "  ${exp_name}: already complete — skipping."
    else
        echo "  Running ${exp_name} …"
        uv run python experiments/run.py "${cfg}"
    fi
done

# ---------------------------------------------------------------------------
# Step 3 — Full ablation sweep
# ---------------------------------------------------------------------------

echo ""
echo "------------------------------------------------------------"
echo "Step 3: Ablation sweep (${ABLATION_SEEDS} seeds each)"
echo "------------------------------------------------------------"

uv run python - <<EOF
from experiments.ablations import AblationRunner, ALL_ABLATIONS
from experiments.configs import load_config

base_cfg = load_config("${BASE_CONFIG}")
runner = AblationRunner(
    base_config=base_cfg,
    ablation_configs=ALL_ABLATIONS,
    output_base_dir="${OUTPUT_DIR}/ablations",
)
runner.run_all(n_seeds=${ABLATION_SEEDS})
print("Ablation sweep complete.")
EOF

# ---------------------------------------------------------------------------
# Step 4 — Generate report (tables, findings, RESULTS.md)
# ---------------------------------------------------------------------------

echo ""
echo "------------------------------------------------------------"
echo "Step 4: Report generation"
echo "------------------------------------------------------------"

# Determine the best experiment directory for UMAP (prefer attention_2layer)
UMAP_MODEL_DIR="${OUTPUT_DIR}/attention_2layer"
if [ ! -f "${UMAP_MODEL_DIR}/model.pt" ]; then
    # Fall back to any experiment directory that has a model
    UMAP_MODEL_DIR=$(find "${OUTPUT_DIR}" -maxdepth 2 -name "model.pt" \
                     | head -1 | xargs dirname 2>/dev/null || echo "")
fi

if [ -n "${UMAP_MODEL_DIR}" ] && [ -f "${UMAP_MODEL_DIR}/model.pt" ]; then
    echo "  Generating report with UMAP from ${UMAP_MODEL_DIR} …"
    uv run python -m experiments.report \
        --results-dir "${OUTPUT_DIR}" \
        --output-dir  "${RESULTS_DIR}" \
        --umap-model-dir "${UMAP_MODEL_DIR}"
else
    echo "  No trained model found; generating report without UMAP …"
    uv run python -m experiments.report \
        --results-dir "${OUTPUT_DIR}" \
        --output-dir  "${RESULTS_DIR}"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

echo ""
echo "========================================================"
echo " Pipeline complete."
echo " Results: ${RESULTS_DIR}/RESULTS.md"
echo " Findings: ${RESULTS_DIR}/FINDINGS.md"
echo "========================================================"
