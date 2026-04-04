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
# INCLUDE_LAZY_BASELINE
#                   When set to 1, include the lazy pairwise baseline
#                   config in Step 1.
#                   Default: 0
#
# The lazy baseline uses the CUDA-capable composite distance path when the
# config requests data.distance_device: cuda.
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
INCLUDE_LAZY_BASELINE="${INCLUDE_LAZY_BASELINE:-0}"

echo "========================================================"
echo " civis — Attribute Embedding Reproducibility Pipeline"
echo "========================================================"
echo "  DATA_DIR       = ${DATA_DIR}"
echo "  OUTPUT_DIR     = ${OUTPUT_DIR}"
echo "  RESULTS_DIR    = ${RESULTS_DIR}"
echo "  ABLATION_SEEDS = ${ABLATION_SEEDS}"
echo "  BASE_CONFIG    = ${BASE_CONFIG}"
echo "  LAZY           = ${INCLUDE_LAZY_BASELINE}"
echo ""

mkdir -p "${OUTPUT_DIR}" "${RESULTS_DIR}"

# ---------------------------------------------------------------------------
# Step 1 — Baseline experiments (one per YAML config)
# Feature caches, candidate indices, and lazy pair-distance caches are
# reused per experiment by the runner.
# ---------------------------------------------------------------------------

echo "------------------------------------------------------------"
echo "Step 1: Baseline experiments"
echo "------------------------------------------------------------"

BASELINE_CONFIGS=(
    "experiments/configs/baseline_addition.yaml"
    "experiments/configs/attention_1layer.yaml"
    "experiments/configs/attention_2layer.yaml"
    "experiments/configs/attention_4layer.yaml"
    "experiments/configs/film.yaml"
)

if [ "${INCLUDE_LAZY_BASELINE}" = "1" ]; then
    BASELINE_CONFIGS+=("experiments/configs/attention_2layer_lazy.yaml")
fi

for cfg in "${BASELINE_CONFIGS[@]}"; do
    # Extract experiment name from the yaml (first `name:` line)
    exp_name=$(grep -m1 '^name:' "${cfg}" | awk '{print $2}')
    marker="${OUTPUT_DIR}/${exp_name}/config.json"

    if [ -f "${marker}" ]; then
        echo "  ${exp_name}: already complete — skipping."
    else
        echo "  Running ${exp_name} …"
        uv run civis run "${cfg}" --data-dir "${DATA_DIR}"
    fi
done

# ---------------------------------------------------------------------------
# Step 2 — Full ablation sweep
# ---------------------------------------------------------------------------

echo ""
echo "------------------------------------------------------------"
echo "Step 2: Ablation sweep (${ABLATION_SEEDS} seeds each)"
echo "------------------------------------------------------------"

uv run civis ablate "${BASE_CONFIG}" \
    --data-dir "${DATA_DIR}" \
    --output-dir "${OUTPUT_DIR}/ablations" \
    --seeds "${ABLATION_SEEDS}"

# ---------------------------------------------------------------------------
# Step 3 — Generate report (tables, findings, RESULTS.md)
# ---------------------------------------------------------------------------

echo ""
echo "------------------------------------------------------------"
echo "Step 3: Report generation"
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
