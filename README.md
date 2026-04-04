# civis

Attribute Embedding Learning for Human Activity Schedules: scalable contrastive learning with schedule-distance supervision.

## Quick Start

```bash
# Install
uv sync

# Run an experiment
uv run civis run experiments/configs/baseline_addition.yaml

# Run validation
uv run pytest tests/

# Ablation sweeps
uv run civis ablate experiments/configs/attention_2layer.yaml --seeds 3
```

---

## Architecture Overview: Distances, Features, and Metrics

All distance metrics use a **scalable V2 API** with three phases:

1. **Feature Extraction**: `prepare_features(activities) → dict`  
    Precompute and cache participation vectors, sequence 2-gram vectors, and time-use matrices once.

2. **Candidate Index** (optional): `build_candidate_index(features) → index`  
   Build approximate nearest-neighbor index for large datasets (O(N log N) construction, not O(N²)).

3. **Batch Scoring**: `score_pairs_batch(features, pairs, index) → distances`  
   Score pairs in efficient vectorized batches without materializing full N×N matrix.

This enables scaling without materialising dense distance matrices. On the
lazy composite path, pair scoring can run on CUDA when available via
``data.distance_device: cuda``.

---

## distances Module: Schedule Distance Metrics

### Precomputing and Caching Features

Features are versioned by content hash and can be reused across multiple distance metrics:

```python
from distances import build_schedule_features, load_schedule_features

# First run: extract and cache
features = build_schedule_features(
    activities_df,
    feature_dir="cache/features",
    timing_resolution=1,  # per-minute
    overwrite=False,      # reuse if exists
)

# Later runs: auto-loads from cache
features = load_schedule_features("cache/features")
print(f"Extracted {len(features.pids)} schedules")
print(f"Data version: {features.manifest.data_hash[:8]}...")
```

**Cached artifacts**:
- `features.npz`: Compressed NumPy archive (pids, participation matrix, timing matrix)
- `sequences.json`: Ordered activity sequences
- `manifest.json`: Metadata (content hash, schema, extraction time)

### Distance Scoring Approach

The training pipeline always uses lazy pairwise composite scoring with three
components: participation, sequence 2-gram, and timing. The scorer is
available as both ``CompositeDistance`` and ``GPUCompositeDistance``.

### Composite Distance: Weighted Combination

Combine participation, sequence, and timing distances:

```python
from distances.metric_plugins import CompositeDistance, ParticipationDistance, TwoGramDistance, TimingDistance

# Build individual metrics
part_metric = ParticipationDistance()
seq_metric = TwoGramDistance()
timing_metric = TimingMetric(resolution=10)

# Composite with weights
composite = CompositeDistance(
    components=[part_metric, seq_metric, timing_metric],
    weights=(0.33, 0.33, 0.34),
    normalize_weights=True,
)

# Extract features (once)
features = composite.prepare_features(activities_df)

# Score pairs in batch
pairs = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int32)
distances = composite.score_pairs_batch(features, pairs)
```

### Metric Plugin Registry

Extend or customize distance metrics via the registry:

```python
from distances import build_metric

# Built-in metrics: "participation", "sequence", "timing", "composite", "composite_gpu"
metric = build_metric("participation")

# Metrics may also expose a candidate index for approximate neighbour lookup
features = metric.prepare_features(activities_df)
index = metric.build_candidate_index(features)

# Config-based composites
metric = build_metric({
    "name": "composite",
    "weights": [0.4, 0.3, 0.3],
    "components": ["participation", "sequence", "timing"]
})

```

**Available Resolution Trade-offs**:
- Timing resolution: `1` (per-minute, 1440 bins) = slow but precise
- Timing resolution: `10` (default, 144 bins) = fast and practical
- Timing resolution: `60` (hourly, 24 bins) = very fast, less precise

**Candidate-index support**:
- `ParticipationMetric` and `TimingMetric` build a lightweight k-NN index over their feature matrices.
- `SequenceMetric` (2-gram vectors) also supports k-NN indexing.
- `CompositeMetric` bundles any component indices that are available.

---

## datasets Module: PyTorch Datasets and Masking

### Attribute Encoding

```python
from datasets.encoding import AttributeEncoder, default_attribute_configs

encoder = AttributeEncoder(default_attribute_configs())
encoder.fit(attributes_df)
encoded_attrs = encoder.transform(attributes_df)

# Output: dict[str, torch.Tensor]
# - discrete: int64, values in [0, vocab_size]
# - continuous: float32, normalized to [0, 1]
```

### Training Datasets with Distance Supervision

```python
from datasets.dataset import ScheduleEmbeddingDataset, SparseDistanceMatrix

# Sparse storage (k-NN neighbors only)
sparse_dist = SparseDistanceMatrix.from_dense(D, k=500)

# Pairwise mode: (attrs_i, attrs_j, distance_ij)
dataset = ScheduleEmbeddingDataset(
    attributes=encoded_attrs,
    distance_matrix=sparse_dist,
    mode="pairwise",
    sampling_strategy="random",  # or "hard_negative"
)

# Triplet mode: (anchor, positive, negative, d_ap, d_an)
dataset = ScheduleEmbeddingDataset(
    attributes=encoded_attrs,
    distance_matrix=sparse_dist,
    mode="triplet",
    positive_threshold=0.2,
    negative_threshold=0.5,
)
```

### Lazy Pairwise Backend (Large Data)

Distances are computed lazily per sampled pair via
metric plugin `score_pairs_batch`.

On this backend, civis still precomputes and caches all schedule features in
`<output>/<run>/feature_store/` and also persists a pair-distance memo cache to
`<output>/<run>/lazy_distance_cache.npz` for reuse across reruns. When CUDA is
available, the lazy scorer can move feature matrices to GPU with
`data.distance_device: cuda`.

```yaml
data:
    mode: pairwise
    distance_device: cuda
    timing_resolution: 10
    distance_weights: { participation: 0.33, sequence: 0.33, timing: 0.34 }
```

```bash
# Built-in lazy config
uv run civis run experiments/configs/attention_2layer_lazy.yaml

# Same config with external data directory override
uv run civis run experiments/configs/attention_2layer_lazy.yaml --data-dir /path/to/data
```

Notes:
- Only `mode: pairwise` is supported.
- Tune cache size with `data.lazy_max_cached_pairs`.
- Use `data.distance_device: cuda` to enable GPU pair scoring when available.

### Attribute Masking

Three masking strategies for robustness:

```python
from datasets.masking import AttributeMasker

masker = AttributeMasker.from_data(
    attributes_df,
    base_rate=0.15,
    missingness_weighted=True,
    strategy="curriculum",  # "independent", "grouped", or "curriculum"
)

# Apply during sampling (not preprocessing)
masked_attrs = masker(attributes)
```

---

## models Module: Embedding Models

All embedders consume attribute dictionaries and output fixed-width embeddings:

```python
from models.addition import AdditionEmbedder
from models.attention import SelfAttentionEmbedder
from models.film import FiLMEmbedder

model = SelfAttentionEmbedder(
    attribute_configs=encoder.configs,
    d_embed=64,
    d_model=128,
    n_heads=4,
    n_layers=2,
)

# Forward pass
embeddings = model(attributes)  # (B, 128)
```

**Model Types**:
- `AdditionEmbedder`: Sum pool over attribute embeddings (baseline)
- `SelfAttentionEmbedder`: Transformer encoder over attribute tokens (recommended)
- `FiLMEmbedder`: FiLM conditioning for feature modulation

---

## training Module: Contrastive Learning

PyTorch Lightning training with multiple loss functions:

```python
from training.losses import DistanceRegressionLoss, SoftNearestNeighbourLoss
from training.trainer import EmbeddingTrainer, TrainerConfig

config = TrainerConfig(
    lr=1e-3,
    max_epochs=100,
    warmup_steps=1000,
)

trainer = EmbeddingTrainer(
    model=model,
    loss_fn=DistanceRegressionLoss(),
    config=config,
)
```

**Loss Functions**:
- `DistanceRegressionLoss`: MSE between predicted and true squared distances
- `SoftNearestNeighbourLoss`: Softmax over batch distances
- `RankCorrelationLoss`: Spearman correlation preservation
- `NTXent`: Contrastive NT-Xent loss

**Optional Training Features**:
- Hard negative mining with periodic k-NN index refresh
- Curriculum attribute masking (increasing difficulty over time)
- Collapse monitoring (prevent embedding space collapse)
- Attention weight logging (for attention models)

---

## evaluation Module: Downstream and Intrinsic Tasks

### Downstream Task Evaluation

```python
from evaluation.discrete import DiscreteDownstreamTask
from evaluation.continuous import ContinuousDownstreamTask

# Freeze embedder, train lightweight head on new task
task = DiscreteDownstreamTask(
    task_name="employment",
    embedding_model=embedder,
    attributes=val_attrs,
    labels=val_labels,
)

metrics = task.run()
# → {"linear/accuracy": 0.87, "mlp/auc": 0.92, ...}
```

### Intrinsic Embedding Geometry Analysis

```python
from evaluation.geometry import GeometryAnalyser

analyser = GeometryAnalyser(embedder)
report = analyser.full_report(
    embeddings=val_emb,
    distances=val_D,
)

# Returns: alignment, uniformity, rank correlation, neighbourhood overlap, source separation, CKA
```

---

## experiments Module: Configuration and Orchestration

### Running Experiments

Config YAML structure:

```yaml
name: baseline_attention
seed: 42

data:
  data_path: data/activities.parquet
  attributes_path: data/attributes.parquet
  distance_weights: { participation: 0.33, sequence: 0.33, timing: 0.34 }
    mode: pairwise
    timing_resolution: 10
    lazy_max_cached_pairs: 500000

model:
  architecture: attention
  d_embed: 64
  d_model: 128

training:
  lr: 0.001
  max_epochs: 100

evaluation:
  downstream_tasks: [employment, income, education]
```

```python
from experiments.run import run_experiment

run_experiment("experiments/configs/baseline_attention.yaml")
```

### Ablation Sweeps

```bash
uv run civis ablate experiments/configs/attention_2layer.yaml --seeds 3
uv run civis ablate experiments/configs/baseline_addition.yaml --seeds 1
```

Generates comparison tables and figures in `experiments/results/`.

---

## CLI: main.py

```bash
# Run a single experiment
uv run python main.py run experiments/configs/baseline_addition.yaml

# Validate data and distances
uv run python main.py validate data/activities.parquet data/attributes.parquet

# Run ablation group
uv run python main.py ablate --group architecture

# Generate comparison report
uv run python main.py report --ablation_dir experiments/ablations/architecture/
```

---

## Data Format

### activities.csv / activities.parquet

Columns: `pid`, `seq`, `act`, `zone`, `start`, `end`
- `pid` (Utf8): Person ID
- `seq` (Int32): Activity sequence number (within person, 0-indexed)
- `act` (Utf8): Activity type (one of 9 types)
- `zone` (Utf8): Spatial zone
- `start` (Int32): Minutes since midnight [0, 1440)
- `end` (Int32): Minutes since midnight (0, 1440]

### attributes.csv / attributes.parquet

Columns: `pid`, `hid`, `age`, `hh_size`, `hh_income`, `sex`, `dwelling`, `ownership`, `vehicles`, `disability`, `education`, `can_wfh`, `occupation`, `race`, `has_licence`, `relationship`, `employment`, `country`, `source`, `year`, `month`, `day`, `hh_zone`, `weight`, `access_egress_distance`, `max_temp_c`, `rain`, `avg_speed`
- Discrete attributes mapped to integer indices (0 = unknown)
- Continuous attributes normalized to [0, 1]

---

## Project Structure

```
civis/
├── distances/          # Schedule distance metrics, features, caching
├── datasets/          # PyTorch datasets, encoding, masking
├── models/            # Embedding models (addition, attention, FiLM)
├── training/          # Losses, PyTorch Lightning trainer
├── evaluation/        # Downstream tasks, intrinsic geometry analysis
├── experiments/       # Configs, ablations, reporting
├── main.py            # CLI
├── tests/             # Unit tests (pytest)
└── data/              # Example datasets (not in repo)
```

---

## Installation and Development

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Format code
uv run ruff check --fix distances/ datasets/ models/ training/ evaluation/

# Type check
uv run pyright distances/
```

---

## References

**Contrastive Learning**: Spectral distribution alignment (Wang & Isola, ICLR 2021)  
**Sequence 2-grams**: Normalized transition-vector distance over activity bigrams  
**Wasserstein Timing**: 1D earth-mover distance for activity timing distributions  
**Feature Hashing**: Content-addressable versioning for reproducibility

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

PyTorch Dataset yielding lazy pairwise contrastive samples.

```python
import numpy as np
from torch.utils.data import DataLoader
from datasets.dataset import LazyPairwiseDataset, collate_fn
from distances.metric_plugins import CompositeDistance, ParticipationDistance, TwoGramDistance, TimingDistance

metric = CompositeDistance(
    components=[ParticipationDistance(), TwoGramDistance(), TimingDistance(resolution=10)],
    weights=(0.33, 0.33, 0.34),
)
features = metric.prepare_features(acts)

ds = LazyPairwiseDataset(
    attributes=encoded,
    global_indices=np.arange(len(next(iter(encoded.values()))), dtype=np.int64),
    metric=metric,
    metric_features=features,
    masker=masker,
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
- Validation metrics: alignment, uniformity, Spearman rank correlation (on fixed held-out pairs)
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

`generate_umap_from_dir` loads `config.json`, `encoder.pkl`, `model.pt`, and `distance_matrix.npz` saved automatically by the training runner and generates three figures under `results/figures/`:

| Figure | Colour |
|--------|--------|
| `umap_source.png` | Data source |
| `umap_employment.png` | Employment status |
| `umap_schedule_cluster.png` | k-means clusters on schedule distance matrix |

## CLI

After `uv sync`, the `civis` command is available with three subcommands:

```bash
# Run a full training experiment from a YAML config
civis run experiments/configs/baseline_addition.yaml

# Validate data files and print summary statistics
civis validate data/activities.parquet data/attributes.parquet

# Run an ablation study from a base config
civis ablate experiments/configs/baseline_addition.yaml
```

Each subcommand accepts `--help` for usage details.

## Reproducibility

The full pipeline (distances → training → evaluation → report) is driven by a single script:

```bash
DATA_DIR=/path/to/data bash scripts/reproduce.sh
```

Optional environment variables: `OUTPUT_DIR` (default `outputs`), `RESULTS_DIR` (default `results`), `ABLATION_SEEDS` (default `3`), `BASE_CONFIG` (default `experiments/configs/attention_2layer.yaml`).

To include the on-the-fly pairwise baseline in Step 1, set:

```bash
INCLUDE_LAZY_BASELINE=1 DATA_DIR=/path/to/data bash scripts/reproduce.sh
```

The script is idempotent — each stage is skipped if its outputs already exist, so interrupted runs resume cleanly.

Dependency files:
- `requirements.txt` — pinned pip packages (`uv export --no-hashes --no-dev`)
- `environment.yml` — conda environment referencing `requirements.txt`

To install with test dependencies: `uv sync --group dev`

To run unit tests: `uv run pytest -m "not slow"`

To run the end-to-end smoke test: `uv run pytest -m slow tests/test_smoke.py`
