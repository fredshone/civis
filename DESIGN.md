# civis Design: Scalable Embedding Learning for Activity Schedules

## Purpose

Contrastive learning framework that embeds heterogeneous human attributes such that embedding geometry reflects empirical scheduling behaviour. Uses three-component distance metrics (participation + sequence + timing) as training supervision.

Designed for **scalability**: feature precomputation, sparse k-NN caching, and V2 metric API enable efficient processing of millions of schedules without dense N×N matrices.

## End-to-End Flow

```
activities + attributes
    ↓
[distances/feature_store] → persist (participation, sequences, timing)
    ↓
[distances/metric_plugins] → CompositeMetric with V2 API
    ↓
[distances/cache] → sparse k-NN graph + manifest
    ↓
[datasets/encoding] → encode attributes → dict[str, Tensor]
    ↓
[datasets/dataset] → sample (attrs_i, attrs_j, distance) pairs from graph
    ↓
[models/*] → embedder(attrs) → Tensor(B, d_model)
    ↓
[training/trainer] → contrastive learning loop
    ↓
[evaluation/*] → downstream + intrinsic tasks
    ↓
[experiments/*] → config-driven orchestration
```

---

## Scalable Metric API (V2)

All distance metrics implement three core methods:

### 1. `prepare_features(activities) → dict[str, Any]`

**Purpose**: Extract and return reusable schedule representations (called once per dataset).

**Output Contract**:
- Always includes `"pids": list[str]` (person identifiers in order)
- Metric-specific features (e.g., matrices, sequences, indices)

**Lifecycle**: Called **once per dataset**, results cached via `FeatureStore` for reuse across experiments.

**Example** (CompositeMetric):
```python
{
    "pids": ["pid_0", "pid_1", ..., "pid_N"],
    "components": [
        {"pids": [...], "matrix": ndarray(N, 9)},      # participation
        {"pids": [...], "sequences": list[list[str]]},  # sequence
        {"pids": [...], "matrix": ndarray(N, T)},       # timing
    ]
}
```

### 2. `build_candidate_index(features) → dict[str, Any] | None`

**Purpose**: Build optional approximate nearest-neighbor index for efficient candidate generation.

**When to Use**:
- Small/medium datasets (< 1M schedules): return None (brute-force in score_pairs_batch)
- Large datasets (> 1M schedules): build FAISS/sklearn NearestNeighbors index
- Construction: O(N log N) vs brute-force scoring: O(M × cost_per_pair) where M = num pairs to score

**Output Contract**:
```python
{
    "type": "faiss" | "sklearn" | "bruteforce",
    "index": <index_object>,
    "nn_k": int,  # number of neighbors stored per person
}
```

**Default**: Returns None (tells score_pairs_batch to use brute-force).

**Current implementation notes**:
- `ParticipationMetric` and `TimingMetric` expose a k-NN index over their dense feature matrices.
- `SequenceMetric` remains brute-force, because no compact vector embedding is defined yet.
- `CompositeMetric` collects any available component indices into a composite bundle for downstream candidate generation.

### 3. `score_pairs_batch(features, pairs, index=None) → ndarray`

**Purpose**: Score a batch of person-pair indices using precomputed features (no materialization of full N×N matrix).

**Input**:
- `features`: dict from prepare_features()
- `pairs`: shape (M, 2) int32 array of (i, j) indices to score
- `index`: optional dict from build_candidate_index()

**Output**: shape (M,) float64 array of distances in [0, 1].

**Complexity**: O(M × cost_per_pair) where M is small (batch size) or moderate (k-nearest neighbors), **not** O(N²).

**Example Implementation** (Participation metric):
```python
def score_pairs_batch(self, features, pairs, index=None):
    matrix = features["matrix"]  # shape (N, 9)
    distances = np.zeros(len(pairs), dtype=np.float64)
    for idx, (i, j) in enumerate(pairs):
        distances[idx] = l1_norm(matrix[i], matrix[j])
    return distances
```

---

## Feature Precomputation and Versioning

### FeatureStore

Separates **extraction** from **scoring** — enables:
- Reuse of expensive feature extraction across multiple metrics
- Deterministic versioning via content hashing (cache invalidation on data change)
- Separation of concerns (feature schema from distance computation)

**API**:
```python
features = build_schedule_features(
    activities,
    feature_dir="cache/features",
    timing_resolution=1,
    overwrite=False,
)

# Artifacts saved:
# - features.npz: numpy archive (pids, participation, timing matrices)
# - sequences.json: activity sequence lists
# - manifest.json: metadata (data hash, schema, timestamp)
```

**Reuse**: Multiple metrics (participation, sequence, timing) and multiple experiments share cached features via manifests.

### DistanceGraph (Sparse k-NN Cache)

For large N, storing full N×N distance matrix is **infeasible** (~128 TB for 4M humans @ float64).

**Solution**: Store only **meaningful neighbors** (training pairs).

**API**:
```python
graph = build_distance_graph(
    pids=...,
    distance_matrix=D,  # computed once, full N×N
    metric_spec=...,
    feature_manifest=...,
    graph_dir="cache/graph",
    k=500,  # neighbors per person
)

# Artifacts saved:
# - graph.npz: sparse edge store (i, j, distance) tuples, O(N·k) not O(N²)
# - manifest.json: links to feature manifest, metric hash

# Retrieval:
d_ij = graph.get_distance(i, j)  # O(log k) lookup or NaN if not neighbor
```

**Memory Scaling**:
- Dense matrix: O(N²) = 128 TB @ 4M × float64
- Sparse graph (k=500): O(N·k) = 8 GB @ 4M × 500 × float64

---

## Interface Contracts by Boundary

### 1. `distances.feature_store` → Any consumer

- Input: raw `activities` DataFrame
- Output: `features` dict (pids, participation, sequences, timing) + `FeatureManifest` (versioning)
- Contract: **Stateless** and **reproducible** (same data → same content hash).
- Caching: Keyed by manifest hash; manual cache invalidation via `overwrite=True`.

### 2. `distances.metric_plugins` → `training` / Any loss function

- Input: `features` dict (from prepare_features) + `pairs` array (M, 2) indices
- Output: shape (M,) float64 array of distances
- Contract: **Batch scoring only**, no full N×N matrix materialization. Metrics call `score_pairs_batch(features, pairs, index)` on demand.
- No state: distance computation is deterministic and stateless given features and pairs.

### 3. `distances.cache` → `datasets.dataset`

- Input: Computed distance matrix + feature manifest + metric spec
- Output: Sparse k-NN graph with O(N·k) memory footprint
- Contract: `get_distance(i, j)` lookup is O(log k) or returns NaN if not neighbors.
- Lifts: Dense matrix burden from memory (128 TB → 8 GB for 4M @ k=500).

### 4. Raw attributes → `datasets.encoding`

- Input: Polars DataFrame from `load_attributes(...)`
- Output: `dict[str, torch.Tensor]` where each tensor has shape `(N,)`.
  - discrete attributes: `torch.int64` (index `0` reserved for unknown/masked)
  - continuous attributes: `torch.float32` normalized to `[0, 1]`
- Contract: This dictionary is the canonical embedder input format across the project.

### 5. `datasets.masking` → `datasets.dataset` → `models`

- `AttributeMasker` mutates attribute values at sampling time (not schema):
  - discrete masking sets value to `0` (unknown index)
  - continuous masking sets value to `0.0`
- `ScheduleEmbeddingDataset` returns mode-specific batches sampled from sparse distance graph:
  - `pairwise`: `(attrs_i, attrs_j, d_ij)`
  - `triplet`: `(anchor, positive, negative)` from mined triplets
  - `single`: `(attrs, distance_row)` for soft-NN loss
- `LazyPairwiseDataset` supports pairwise mode without a precomputed dense matrix,
    computing `d_ij` lazily through `DistanceMetric.score_pairs_batch`.
- `collate_fn` stacks per-sample dictionaries into batch dictionaries keyed by attribute name.

### 6. `datasets` → `models`

- All embedders implement `BaseAttributeEmbedder` and consume `dict[str, Tensor]` batches.
- Shared output contract: embedding tensor of shape `(B, d_model)`.
- No architecture-specific assumptions about attribute order or schema.

### 7. `models` → `training.losses`

- Standard loss contract: `(emb_i, emb_j, distances) → (loss, diagnostics)`.
- Special contract for soft nearest neighbour: `(emb, dist_matrix_bxb) → (loss, diagnostics)`.
- Loss diagnostics are scalar metric dicts logged by `EmbeddingTrainer`.

### 8. `datasets` + `models` + `losses` → `training.trainer`

- `EmbeddingTrainer` (PyTorch Lightning) orchestrates training loop.
- Optional callbacks for:
  - `HardNegativeSampler`: periodically rebuild k-NN index in embedding space
  - `AttributeMasker.set_step(...)`: curriculum masking schedule
  - `loss_fn.set_step(...)`: learning rate / temperature annealing
  - Collapse monitoring: variance ratio checks
  - Attention logging: visualize transformer patterns

### 9. Trained embedder → `evaluation`

- Evaluators receive frozen embedder as callable: `embedder(attrs_dict) → embeddings`.
- Downstream evaluators train lightweight task heads on frozen embeddings (classification/regression).
- Intrinsic evaluators inspect geometry (alignment, uniformity, rank correlation, source separation).

### 10. `experiments` as integration layer

- `experiments/configs.py`: Strongly-typed run configuration (data, model, training, eval sections).
- `experiments/run.py`: Primary composition point for all module interfaces.
- `experiments/ablations.py`: Multi-seed grid sweeps over config deltas.
- `experiments/report.py`: Comparison tables and figures.

---

## Extension Points

1. **Distance Metrics**: Subclass `DistanceMetric` (from `distances/protocols.py`), implement `prepare_features()` and `score_pairs_batch()`, register in `distances/registry.py`.
2. **Models**: Subclass `BaseAttributeEmbedder`, register in `models/registry.py`.
3. **Losses**: Add to `training/losses.py` and register in `LOSS_REGISTRY`.
4. **Evaluators**: Subclass `DownstreamEvaluator`, add config in `evaluation/*.py` task-specific modules.
5. **Ablation Groups**: Define metric collections in `experiments/report.py` `_GROUP_METRICS`.

---

## Core Invariants

1. **Person index alignment**: Preserved across encoded attributes, distance features, and embedder outputs via pids list.
2. **Unknown handling**: Consistent `0` (discrete) / `0.0` (continuous) from masking through all layers.
3. **Fixed embedding width**: All embedders output `(B, d_model)` vectors regardless of architecture.
4. **Deterministic versioning**: Same data → same feature manifest hash → cache reuse.
5. **No dense matrices in training**: Distance sampling uses sparse k-NN graph or batch scoring (O(N·k)), never full O(N²) matrix.
6. **Metric API v2 only**: All metrics implement `prepare_features` + `score_pairs_batch`, never `prepare` + `pair` (v1 removed).
7. **Approximate candidate generation is optional**: when available, metrics can expose a candidate index, but exact batch scoring remains the source of truth.

---

## Performance Considerations

### Computation Bottlenecks

1. **Feature Extraction**: O(N) for participation + O(N) for sequences + O(N) for timing. Typically < 1 min for 1M people on CPU.
2. **Distance Computation**: O(N² × cost_per_pair). Sequence edit distance dominates (~O(N²·seq_len²) without approximations).
3. **Embedder Training**: O(epochs × batch_size × model_forward). Transformer attention is O(seq_len²) per sample.

### Memory Bottlenecks

1. **Dense Distance Matrix**: O(N²) float64 → 128 TB for 4M schedules. **Mitigated**: Use sparse k-NN graph (O(N·k) ≈ 8 GB).
2. **Feature Extraction**: O(N × seq_len) for sequences, O(N × 1440) for per-minute timing. **Solution**: Lazy loading, mmap.
3. **Batch Training**: O(batch_size × model_params). Standard PyTorch solutions (gradient checkpointing, mixed precision).

### Scaling Strategies

- **Small (< 100k schedules)**: Dense matrix + CPU, single machine.
- **Medium (100k–10M)**: Sparse k-NN graph + GPU, single machine with 128+ GB RAM.
- **Large (> 10M)**: Distributed distance computation + FAISS ANN + multi-node training (Ray/torch.distributed).

---

## Testing Strategy

- **Unit tests** (pytest): Individual functions in each module `tests/test_*.py`.
- **Integration tests**: End-to-end pipeline with fixture data (3 sample schedules in `tests/fixtures/`).
- **Validation tests** (distances/VALIDATION.md): Qualitative distribution checks on real data.
- **Ablation tests** (experiments/ablations.py): Systematic comparison of architectural choices.

---

## Key Assumptions

1. Activity schedules are **24-hour day cycles** ([0, 1440) minutes).
2. **Nine activity types** (home, work, education, leisure, medical, escort, other, visit, shop).
3. Attributes are **fairly dense** (< 30% missingness after encoding).
4. **Contrastive signal** is strong: schedule distance correlates with attribute similarity.
5. **Embeddings** serve downstream supervised tasks (not solely clustering/retrieval).
6. **Feature schema** is stable across runs (new attributes require re-extraction).
