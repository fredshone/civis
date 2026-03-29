# Distance Function Validation

Validation of the three component distances and the composite distance for
Attribute Embedding Learning on Human Activity Schedules.

---

## Components

| Component | Module | Range | Metric |
|-----------|--------|-------|--------|
| Participation | `distances.participation` | [0, 1] | L1 / total variation |
| Sequence | `distances.sequence` | [0, 1] | Normalised edit distance |
| Timing | `distances.timing` | [0, 1] | Normalised Hamming |
| Composite | `distances.composite` | [0, 1] | Weighted sum |

---

## Validation Protocol

### A. Component distance distributions

Sample 1 000 random schedule pairs.  For each pair compute all three
component distances and the composite.  Plot:

1. Histograms of each component — check that distributions are not
   degenerate (collapsed near 0 or 1).
2. Pairwise scatter plots of all three components against each other —
   check that they capture partially independent variation rather than
   being collinear.

**Expected:** Unimodal or slightly right-skewed distributions centred in
[0.2, 0.7].  Moderate positive correlation between sequence and timing
distances (both penalise activity-type mismatches), but participation and
timing should be less correlated since timing additionally penalises
temporal placement.

---

### B. Qualitative plausibility — most similar and dissimilar pairs

Find the 10 most similar and 10 most dissimilar pairs by composite distance
and plot each pair as side-by-side horizontal Gantt charts.

**Expected:**
- Most similar pairs should look nearly identical — same activity types in
  roughly the same order and time slots.
- Most dissimilar pairs should have very different activity mixes and
  timing patterns (e.g. a complex work-day schedule vs. a simple home-only
  schedule).

---

### C. Distribution coverage

Compute the full pairwise distance matrix (or a stratified 5 000-pair
sample for large datasets).  Plot the empirical CDF of composite distances.

**Expected:** CDF should be roughly S-shaped, covering most of [0.1, 0.9].
If more than 20 % of pairs fall below 0.05 or above 0.95, investigate
whether the distance is poorly calibrated.

---

### D. Group-level discriminability

Stratify schedules by two binary attributes:

- **Employment status** (employed vs. not employed)
- **Day type** (weekday vs. weekend)

For each attribute compute mean within-group composite distance and
mean between-group composite distance.

**Expected:** Within-group distance < between-group distance for both
attributes, confirming that the composite distance captures variation that
is meaningful at the population level.

---

## Results

> **Note:** This section is populated once real schedule data is available.
> Run `python -m distances.composite --validate data/activities.parquet`
> to regenerate.

### A. Distribution summary

| Component | Mean | Std | Min | Max |
|-----------|------|-----|-----|-----|
| Participation | — | — | — | — |
| Sequence | — | — | — | — |
| Timing | — | — | — | — |
| Composite | — | — | — | — |

### B. Qualitative check

*Plots saved to `distances/figures/` once generated.*

### C. CDF coverage

| Percentile | Composite distance |
|------------|--------------------|
| 10th | — |
| 25th | — |
| 50th | — |
| 75th | — |
| 90th | — |

### D. Group discriminability

| Attribute | Within-group mean | Between-group mean | Ratio |
|-----------|-------------------|--------------------|-------|
| Employment | — | — | — |
| Day type | — | — | — |
