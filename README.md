# civis

Attribute Embedding Learning for Human Activity Schedules.

## distances.data

Loading and array extraction for activity schedule data.

```python
from distances.data import (
    load_activities, load_attributes,
    participation_matrix, time_use_matrix, activity_sequences,
)

acts  = load_activities("data/activities.csv")
attrs = load_attributes("data/attributes.csv")

# (N, 9) float64 — fraction of day spent in each activity type, rows sum to 1
pids, part = participation_matrix(acts)

# (N, 144) int32 — majority activity-type index per 10-min bin (default)
pids, tuse = time_use_matrix(acts)
pids, tuse = time_use_matrix(acts, resolution=5)   # (N, 288) bins

# list of ordered activity-type string lists
pids, seqs = activity_sequences(acts)
```

Activity types are indexed by `ACTIVITY_TYPES`:
`home, work, education, leisure, medical, escort, other, visit, shop`.

For exploration, `python -m distances.data [activities.csv] [attributes.csv]`
prints summary statistics.

## distances.sequence

Normalised edit distance between activity-type sequences.

```python
from distances.sequence import edit_distance, pairwise_sequence_distance, DEFAULT_COST_MATRIX

# Unit costs (every substitution = 1, normalised by max sequence length)
d = edit_distance(["home", "work", "home"], ["home", "education", "leisure", "home"])

# Semantic costs — work↔education, leisure↔visit, etc. cost 0.5
d = edit_distance(seq1, seq2, cost_matrix=DEFAULT_COST_MATRIX)

# Pairwise matrix for a list of sequences (parallelised with joblib)
D = pairwise_sequence_distance(sequences, cost_matrix=DEFAULT_COST_MATRIX, n_jobs=-1)
```

## distances.timing

Timing distances capturing *when* activities happen.

```python
from distances.timing import (
    timing_distance, activity_timing_distance,
    pairwise_timing_distance, pairwise_activity_timing_distance,
)
from distances.data import time_use_matrix

# (N, 1440) int32 — majority activity-type index per minute
pids, tuse = time_use_matrix(acts, resolution=1)

# Hamming: fraction of minutes that disagree, in [0, 1]
d = timing_distance(tuse[0], tuse[1])
D = pairwise_timing_distance(tuse)

# Wasserstein: earth-mover distance for one activity type's daily timing
d = activity_timing_distance(tuse[0], tuse[1], activity_type_idx=1)  # 1 = work
D = pairwise_activity_timing_distance(tuse, activity_type_idx=1, n_jobs=-1)
```

## distances.composite

Weighted combination of all three distance components.

```python
from distances.composite import composite_distance, pairwise_composite_distance

# Scalar composite for a single pair
d = composite_distance(part1, part2, seq1, seq2, time1, time2, weights=(1/3, 1/3, 1/3))

# Full pairwise matrix (parallelised; optional disk cache)
pids, D = pairwise_composite_distance(
    acts,
    weights=(0.4, 0.3, 0.3),
    n_jobs=-1,
    cache_path="experiments/distances.npz",
)
```

## datasets

PyTorch dataset and dataloader for contrastive training.

### datasets.encoding

Encodes raw attribute DataFrames into tensors.

```python
from distances.data import load_attributes
from datasets.encoding import AttributeEncoder, default_attribute_configs

attrs = load_attributes("data/attributes.csv")
enc = AttributeEncoder(default_attribute_configs()).fit(attrs)

# dict[str, Tensor] — discrete attrs as int64 (0 = unknown), continuous as float32 in [0, 1]
encoded = enc.transform(attrs)

enc.save("experiments/encoder.pkl")
enc = AttributeEncoder.load("experiments/encoder.pkl")
```

### datasets.masking

Attribute-level dropout — randomly replaces attribute values with the unknown token (0) during training.

```python
from datasets.masking import AttributeMasker

# Probabilities proportional to empirical missingness, mean rate = 15%
masker = AttributeMasker.from_data(attrs, base_rate=0.15)

# Three strategies: "independent" (default), "grouped", "curriculum"
masker = AttributeMasker(
    mask_probs={"age": 0.2, "employment": 0.1},
    strategy="grouped",
    groups={"person": ["age", "sex", "employment"], "household": ["hh_size", "hh_income"]},
    protected=["source"],   # never masked
)
masked_attrs = masker(encoded_attrs)   # returns a new dict, input unchanged

# Curriculum: ramp probability from 0 → target over warmup_steps
masker = AttributeMasker({"age": 0.3}, strategy="curriculum", warmup_steps=10_000)
masker.set_step(training_step)
```

### datasets.dataset

PyTorch Dataset yielding contrastive pairs or triplets.

```python
import numpy as np
from torch.utils.data import DataLoader
from datasets.dataset import ScheduleEmbeddingDataset, SparseDistanceMatrix, collate_fn

# Precompute distances (from distances.composite)
pids, D = pairwise_composite_distance(acts)

# For large datasets, compress to k nearest + k furthest neighbours per person
sparse_D = SparseDistanceMatrix.from_dense(D, k=50)

# Pairwise mode: yields (attrs_i, attrs_j, distance_ij)
ds = ScheduleEmbeddingDataset(encoded, sparse_D, masker=masker, mode="pairwise")

# Triplet mode: yields (anchor, positive, negative, d_ap, d_an)
ds = ScheduleEmbeddingDataset(
    encoded, sparse_D, masker=masker,
    mode="triplet", positive_threshold=0.2, negative_threshold=0.5,
)

dl = DataLoader(ds, batch_size=256, collate_fn=collate_fn, num_workers=4)
```

## models

Attribute embedding model architectures. All models accept a `dict[str, Tensor]` from `AttributeEncoder.transform` and return a `(batch, d_model)` embedding.

```python
from datasets.encoding import AttributeEncoder, default_attribute_configs
from models import AdditionEmbedder, AttributeEmbedderConfig, build_model

enc = AttributeEncoder(default_attribute_configs()).fit(attrs)
config = AttributeEmbedderConfig.from_encoder(enc, d_embed=64, d_model=128)
```

### models.addition

Sum-pooling baseline. Embeds each attribute independently, sums, then projects to `d_model`.

```python
model = AdditionEmbedder(config)
embeddings = model(enc.transform(attrs))   # (N, 128)
```

### models.attention

Transformer encoder. Learns pairwise attribute interactions before pooling. Supports a learned `[CLS]` token, mean pooling, and learned attribute-type group positional encodings.

```python
from models import SelfAttentionEmbedder

config = AttributeEmbedderConfig.from_encoder(
    enc, d_embed=64, d_model=128,
    n_heads=4, n_layers=2,
    use_cls_token=True, pooling="cls",
    attribute_groups={"sex": "person", "source": "context", "day": "day"},
)
model = SelfAttentionEmbedder(config)
embeddings, attn_weights = model(attrs, return_attention=True)
```

### models.film

FiLM-conditioned model. Designated context attributes (e.g. source, country) modulate content attribute embeddings via learned gamma/beta scale-and-shift. Degrades gracefully to addition when context is fully masked.

```python
from models import FiLMEmbedder

config = AttributeEmbedderConfig.from_encoder(
    enc, d_embed=64, d_model=128,
    context_attributes=["source", "country", "year"],
)
model = FiLMEmbedder(config)
embeddings = model(attrs)
print(model.film_stats())   # {'mean_gamma_deviation': ..., 'mean_beta_magnitude': ...}
```

### models.registry

Config-driven factory and inspection utilities.

```python
from models import build_model, count_parameters, model_summary

model = build_model({"architecture": "attention", "d_embed": 64, "d_model": 128, ...})
print(count_parameters(model))   # {'total': ..., 'trainable': ..., 'per_component': {...}}
model_summary(model, sample_batch)
```

## training

PyTorch Lightning training loop for contrastive embedding learning.

### training.losses

Four loss functions for learning from continuous schedule distances. All return `(loss, diagnostics_dict)`.

```python
from training.losses import build_loss

# MSE between pairwise embedding distances and schedule distances
loss_fn = build_loss({"name": "distance_regression"})
loss_fn = build_loss({"name": "distance_regression", "use_huber": True})

# Differentiable Spearman rank correlation
loss_fn = build_loss({"name": "rank_correlation"})

# Soft nearest-neighbour (requires mode="single" dataset; emb + (B,B) dist matrix)
loss_fn = build_loss({"name": "soft_nearest_neighbour", "tau_schedule": 0.5, "learnable_tau": True})

# NT-Xent with threshold-based positives (multi-positive generalisation)
loss_fn = build_loss({"name": "ntxent", "tau": 0.07, "positive_threshold": 0.2})

loss, diag = loss_fn(emb_i, emb_j, distances)  # diag has per-batch metrics
```

### training.trainer

```python
from training.trainer import EmbeddingTrainer, TrainerConfig, EmbeddingCheckpoint, CollapseMonitor

cfg = TrainerConfig(lr=1e-3, max_epochs=100, warmup_steps=1000)
trainer = EmbeddingTrainer(model, loss_fn, cfg, val_pairs=(attrs_i, attrs_j, distances))
```

`EmbeddingTrainer` is a `LightningModule` with:
- Cosine LR schedule with linear warmup
- Validation metrics: alignment, uniformity, Spearman rank correlation (on fixed held-out pairs), neighbourhood overlap at k=5 and k=20
- Periodic hard-negative k-NN index refresh
- Curriculum masker step updates

Callbacks: `EmbeddingCheckpoint` (saves best by val rank correlation), `CollapseMonitor` (alerts if between-source variance dominates), `AttentionLogger` (logs per-layer attention heatmaps to TensorBoard).

## evaluation

Downstream evaluation tasks for assessing embedding quality. All tasks freeze the pre-trained embedder, compute embeddings once, then train a lightweight sklearn head.

```python
from evaluation import (
    WorkParticipationEvaluator, WorkParticipationConfig,
    WorkDurationEvaluator, WorkDurationConfig,
    TripCountEvaluator, TripCountConfig,
)

evaluator = WorkParticipationEvaluator(model, WorkParticipationConfig())
metrics = evaluator.run(
    train_activities, train_attributes,
    test_activities, test_attributes,
    encoder, train_pids, test_pids,
)
# {'linear/accuracy': ..., 'linear/auc': ..., 'mlp/accuracy': ..., ...}

# Stratified by source or employment status
evaluator.evaluate_stratified(test_emb, test_labels, test_attributes, test_pids, "source")

# Cross-source transfer: train on NTS, test on KTDB
evaluator.cross_source_evaluate(all_acts, all_attrs, encoder, "nts", "ktdb")
```

Three tasks are implemented:

| Class | Task | Head | Metrics |
|-------|------|------|---------|
| `WorkParticipationEvaluator` | Binary: did this person go to work? | Logistic / MLP | accuracy, AUC, F1, Brier |
| `WorkDurationEvaluator` | Regression: total work time (minutes) | Ridge / MLP | MAE, RMSE, R², Spearman |
| `TripCountEvaluator` | Regression: number of trips | Ridge / MLP | MAE, RMSE, R², Spearman |

Baselines for comparison:

```python
from evaluation import random_baseline, frozen_attribute_baseline, compare_embeddings

# Lower bound: Gaussian noise embeddings
random_baseline(evaluator, test_acts, test_attrs, test_pids, embed_dim=128)

# Upper bound: raw one-hot + continuous features, no learned embedding
frozen_attribute_baseline(evaluator, train_acts, train_attrs, test_acts, test_attrs, encoder, train_pids, test_pids)

# Compare multiple embedders on the same task
compare_embeddings({"addition": m1, "attention": m2}, evaluator, ...)  # returns pl.DataFrame
```

ActVAE integration (Task 5.4) — the `CaveatAdapter` wraps any embedder to satisfy the ActVAE label encoder interface. `GenerativeEvaluator.fit()` raises `NotImplementedError` pending ActVAE integration; metric computation is fully implemented.

### Intrinsic evaluation

`GeometryAnalyser` and `AttentionAnalyser` assess embedding quality without downstream task labels.

```python
from evaluation import GeometryAnalyser, GeometryAnalyserConfig

analyser = GeometryAnalyser(
    embedder=model,
    distance_fn=composite_distance,   # schedule distance callable
    test_attributes=test_attrs,
    test_pids=test_pids,
    encoder=encoder,
    masker=masker,   # optional; required for alignment metric
    config=GeometryAnalyserConfig(seed=42),
)

# Individual metrics
au   = analyser.alignment_uniformity()  # {'alignment': float, 'uniformity': float}
rho  = analyser.rank_correlation(n_pairs=5000, schedule_distances=precomputed)
no   = analyser.neighbourhood_overlap(k_values=[5, 10, 20, 50], schedule_distance_matrix=D)
sep  = analyser.source_separation()    # {'mean_wasserstein': ..., 'source_accuracy': ..., ...}
cka  = analyser.cka_with_schedule_kernel(n_samples=500, schedule_distance_matrix=D)

# Full report — runs all metrics, generates plots, writes geometry_report.md
results = analyser.full_report(output_dir="experiments/geometry/")
```

`AttentionAnalyser` is specific to `SelfAttentionEmbedder` and exposes learned attention patterns.

```python
from evaluation import AttentionAnalyser, AttentionAnalyserConfig

aa = AttentionAnalyser(model, test_attrs, test_pids, encoder)

attn = aa.mean_attention_weights()            # (n_layers, seq_len, seq_len)
fig  = aa.plot_attention_heatmap(layer=0)     # labelled heatmap with top-k highlights
imp  = aa.attribute_importance("source")      # which attributes attend most to source
mod  = aa.source_modulation_analysis()        # per-attribute shift when source changes
cons = aa.interaction_consistency()           # {'consistency_score': ..., ...}
```

```python
from evaluation import CaveatAdapter, CaveatAdapterConfig

adapter = CaveatAdapter(model, CaveatAdapterConfig(transfer_mode="frozen"))  # or "fine_tuned", "random_init"
adapter.encode(attrs)   # (batch, d_model) — satisfies LabelEncoderProtocol
```

## experiments

YAML-driven experiment configuration and runner.

```python
from experiments.configs import load_config
from experiments.run import run_experiment

cfg = load_config("experiments/configs/baseline_addition.yaml")
run_experiment("experiments/configs/attention_2layer.yaml")
```

Configs in `experiments/configs/`:

| File | Architecture | Loss |
|------|-------------|------|
| `baseline_addition.yaml` | addition | distance_regression |
| `attention_1layer.yaml` | attention (1L) | distance_regression |
| `attention_2layer.yaml` | attention (2L) | distance_regression |
| `attention_4layer.yaml` | attention (4L) | distance_regression |
| `film.yaml` | FiLM | distance_regression |
| `ablation_loss_regression.yaml` | addition | distance_regression (explicit) |
| `ablation_loss_softnn.yaml` | addition | soft_nearest_neighbour |
| `ablation_no_masking.yaml` | addition | distance_regression, no masking |
| `ablation_masking_high.yaml` | addition | distance_regression, 40% masking |

### experiments.ablations

Systematic ablation runner. Sweeps all planned ablation groups with multiple random seeds, persists per-seed results to JSON, and aggregates into comparison tables and plots.

```python
from experiments.configs import load_config
from experiments.ablations import AblationRunner, ARCHITECTURE_ABLATIONS, ALL_ABLATIONS

base = load_config("experiments/configs/baseline_addition.yaml")

runner = AblationRunner(base, ALL_ABLATIONS, output_base_dir="outputs/ablations")
runner.run_all(n_seeds=3)          # skips completed runs — safe to re-run

# Aggregate mean ± std across seeds
agg = runner.aggregate_results()   # dict[ablation_name, dict[metric, (mean, std)]]

# Paper-ready comparison table (pd.DataFrame, cells = "mean ± std")
table = runner.comparison_table([
    "intrinsic/rank_correlation",
    "intrinsic/neighbourhood_overlap_k10",
    "discrete/work_participation_linear_auc",
    "continuous/work_duration_linear_r2",
])
print(table.to_string())

# Multi-panel bar chart saved to outputs/ablations/ablation_plots.png
runner.plot_ablation_results(["intrinsic/rank_correlation", "discrete/work_participation_linear_auc"])
```

Four pre-defined ablation groups (21 configs total):

| Group | Constant | Variants |
|-------|----------|---------|
| Architecture | `ARCHITECTURE_ABLATIONS` | addition, FiLM, attention 1L/1H, 1L/4H, 2L/4H, 4L/8H |
| Loss | `LOSS_ABLATIONS` | regression (MSE/Huber), soft-NN (fixed/learned τ), rank correlation, NT-Xent |
| Masking | `MASKING_ABLATIONS` | none, 5%, 15%, 30%, grouped, curriculum |
| Embedding dims | `EMBEDDING_DIM_ABLATIONS` | (32,64), (64,128), (128,256) |

Custom ablations can be defined as override dicts and passed to `AblationRunner` directly:

```python
my_ablations = [
    {"name": "my_variant", "model": {"architecture": "attention", "n_layers": 3}},
]
runner = AblationRunner(base, my_ablations, "outputs/custom")
```

Results are stored under `<output_base_dir>/<ablation_name>/seed_<n>/results.json`. Each metrics dict contains `meta/`, `intrinsic/`, `discrete/`, and `continuous/` prefixed keys from the full evaluation pipeline.

## experiments.report

Report generation and UMAP visualisation.  Reads ablation JSON files and trained model artefacts to produce `RESULTS.md`, `FINDINGS.md`, per-group comparison tables, and UMAP scatter plots — without re-running any experiments.

```python
from experiments.report import (
    generate_report,           # assemble master RESULTS.md
    generate_results_tables,   # per-group ablation tables as markdown
    generate_findings,         # FINDINGS.md summarising key results
    generate_umap_from_dir,    # UMAP plots from a saved model directory
    seed_everything,           # seed torch / numpy / random
)

# Tables + FINDINGS.md + RESULTS.md from ablation outputs
generate_report("outputs/", "results/")

# UMAP projections of test-set embeddings (requires outputs/{name}/model.pt etc.)
generate_umap_from_dir("outputs/attention_2layer", "results/")
```

`generate_umap_from_dir` loads `config.json`, `encoder.pkl`, `model.pt`, and `distance_matrix.npy` saved automatically by the training runner and generates three figures under `results/figures/`:

| Figure | Colour |
|--------|--------|
| `umap_source.png` | Data source |
| `umap_employment.png` | Employment status |
| `umap_schedule_cluster.png` | k-means clusters on schedule distance matrix |

## Reproducibility

The full pipeline (distances → training → evaluation → report) is driven by a single script:

```bash
DATA_DIR=/path/to/data bash scripts/reproduce.sh
```

Optional environment variables: `OUTPUT_DIR` (default `outputs`), `RESULTS_DIR` (default `results`), `ABLATION_SEEDS` (default `3`), `BASE_CONFIG` (default `experiments/configs/attention_2layer.yaml`).

The script is idempotent — each stage is skipped if its outputs already exist, so interrupted runs resume cleanly.

Dependency files:
- `requirements.txt` — pinned pip packages (`uv export --no-hashes --no-dev`)
- `environment.yml` — conda environment referencing `requirements.txt`
