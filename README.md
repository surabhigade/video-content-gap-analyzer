# Video Content Gap Analyzer · v2

**Find what's missing in your video dataset before you waste time collecting more of what you already have.**

A [FiftyOne](https://docs.voxel51.com/) plugin that identifies coverage gaps in video datasets using [Twelve Labs](https://twelvelabs.io/) Marengo embeddings, density-based clustering, and on-demand Pegasus descriptions.

## The Problem

Most teams improve their video models by collecting *more* data. But more data doesn't help if it's more of the same. The [CV4Smalls](https://cv4smalls.netlify.app/) workshop series demonstrated that **data curation beats model complexity** — the winning EAR 2025 submission used a model architecture from 2019 and won by picking better training data. Before you collect another 10,000 videos, you should know what your dataset is actually missing.

## What This Plugin Does

1. **Embeds** your video dataset into 512-d vectors using Twelve Labs Marengo 3.0 (cached on disk, keyed by file content — re-runs are ~instant).
2. **Clusters** videos with HDBSCAN by default (density-based, auto-tunes *k*, flags noise natively) or KMeans on demand.
3. **Describes** each cluster on demand via Twelve Labs Pegasus 1.2 — clicking a point fires one API call, caches the result, and tells you what's *common* in that cluster vs. what *varies* within it.
4. **Detects gaps** at three levels: sparse clusters, hierarchical expected-category gaps (parent → child), and priority-ranked overall gap list with concrete collection recommendations.
5. **Visualizes** everything in an interactive FiftyOne panel — KDE heatmap behind the scatter, red diamonds for missing categories projected into UMAP space, consistent 4-bucket health colouring across the plot and report.
6. **Exports** a self-contained HTML report (inline SVG + CSV download) so colleagues without FiftyOne can read and share the results.
7. **Tracks history** — each run appends to `dataset.info["coverage_history"]`, and subsequent reports show a "Since last run" diff at the top.

## Quick Start

### Prerequisites

- Python 3.10+
- A [Twelve Labs](https://twelvelabs.io/) API key (free tier provides 600 minutes of indexing)

### 1. Install the plugin

```bash
git clone https://github.com/surabhigade/video-content-gap-analyzer.git
cd video-content-gap-analyzer

# Install Python dependencies
pip install -r requirements.txt

# Register as a FiftyOne plugin (symlink so edits are picked up live)
ln -s "$(pwd)" "$HOME/fiftyone/__plugins__/video-content-gap-analyzer"
```

If you're on Linux/macOS without cmake, you may need `brew install cmake` or `apt install cmake` before installing `llvmlite` (a transitive dep of `umap-learn`). The existing `requirements.txt` pins compatible `llvmlite<0.45` + `numba<0.62` versions.

**Optional: parametric UMAP** (neural-network-based reducer, enables true incremental updates via `reducer.transform()`):

```bash
pip install 'umap-learn[parametric_umap]'   # pulls in TensorFlow (~500 MB)
```

Without this extra, the plugin uses regular UMAP — the caching still works via pickle and `transform()` (approximate nearest-neighbour), just without the neural-network path.

### 2. Set your API key

```bash
export TWELVELABS_API_KEY="your_key_here"
```

(Or drop it in a `.env` file — already gitignored.)

### 3. Run the demo

```bash
python demo.py
```

Loads 40 videos from [Voxel51/Safe_and_Unsafe_Behaviours](https://huggingface.co/datasets/Voxel51/Safe_and_Unsafe_Behaviours), launches the FiftyOne App, and prints a step-by-step walkthrough in your terminal.

### 4. Use with your own dataset

```python
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

# Load a dataset from HuggingFace Hub…
dataset = load_from_hub("Voxel51/Safe_and_Unsafe_Behaviours", max_samples=40)
# …or load your own video dataset
# dataset = fo.Dataset.from_dir("/path/to/videos", fo.types.VideoDirectory)

session = fo.launch_app(dataset)
```

### 5. Run the analysis

1. Press `` ` `` in the FiftyOne App to open the operator menu.
2. Search for **Analyze Coverage** and select it.
3. Configure parameters (see [Configuration](#configuration) below).
4. Click **Execute** and watch the 4-stage progress bar.

### 6. View the results

- **Coverage Map panel** — click the `+` tab in the panel bar → **Coverage Map**. You'll see:
  - A warm-cool 2-D density heatmap behind the points
  - Scatter points colour-coded by cluster health (🟢 well-covered / 🟡 thin / 🔴 sparse / ⚫ noise)
  - Red hollow diamonds for missing expected categories, projected into UMAP space — they land *in* the sparse zones of the heatmap
  - Click any cluster point to lazy-load its Pegasus description
- **Show Gap Report** — operator → typed panel with three tiers:
  - **Tier 1 · Executive Summary**: severity-coded coverage banner (Error/Warning/Success based on %), totals, one actionable recommendation
  - **Tier 2 · Priority Gaps**: ranked table with severity (Critical/Moderate/Low), priority score, concrete collection recommendation
  - **Tier 3 · Full Breakdown**: per-cluster stats + well-covered categories that need no action
- **Export Coverage Report** — operator → writes a single-file HTML report (inline SVG + CSV download) to a path you choose. Share it with anyone.
- **Filter broken samples** — samples that failed to embed after retries carry an `embedding_error` field. `ds.match(F("embedding_error") != "")` lists them.

## Configuration

Operator parameters (all accessible in the UI when you run **Analyze Coverage**):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_clusters` | `0` | `0` = HDBSCAN (density-based; discovers clusters + flags noise automatically). A positive integer falls back to KMeans with that exact *k* and the classic distance-threshold outlier heuristic. |
| `expected_categories` | `""` | Categories to check for. **Hierarchical format**: `"parent: child1, child2 \| parent2: child3"`. A flat comma-separated list still works (each leaf becomes its own parent). Example: `"forklift operations: forward, reverse, loading \| fall hazards: ladder climb, elevated platform"`. |
| `use_pegasus` | `False` | **Fast mode** (default): skip upfront Pegasus calls; clusters get `"Cluster N"` placeholder labels. Click any cluster in the Coverage Map to lazy-load its description (one API call, session-cached). **Full mode** (`True`): run Pegasus on 4 reps per cluster during the analysis — all descriptions ready for the exported HTML report. |
| `max_samples` | `0` | Cap on samples to process (`0` = all). Useful for limiting API cost on large datasets. |
| `outlier_threshold` | `2.0` | Std deviations above mean centroid distance to flag outliers. Only applied when `num_clusters > 0` (KMeans). Ignored under HDBSCAN, which labels noise natively. |
| `gap_threshold` | `0.3` | Cosine similarity below which a category leaf is flagged as missing. Higher = more gaps. |
| `clear_cache` | `False` | Wipe the persistent embedding cache (`~/.video_gap_analyzer/embeddings.db`) **and** every sample's `embedding` field before running. Forces a fresh Marengo call for every video — useful when the Marengo model updates or you want to reproduce from scratch. |

## Fields written to samples

After `Analyze Coverage` runs, each sample carries:

| Field | Type | Description |
|-------|------|-------------|
| `embedding` | `list[float]` | 512-d Marengo video embedding (cached — re-runs reuse). |
| `cluster_id` | `int` | Cluster assignment. `-1` under HDBSCAN means noise / outlier. |
| `cluster_label` | `str` | Pegasus description (or `"Cluster N"` placeholder in fast mode until clicked). |
| `cluster_diversity` | `str` | **v2** — Pegasus VARIATION note: what *differs* within the cluster vs. what's common (from a diverse 4-sample mix: 2 nearest-centroid + 1 boundary + 1 random). |
| `cluster_confidence` | `float` | **v2** — HDBSCAN membership probability in `[0, 1]`. Low-but-non-zero values mark cluster-boundary points — a soft gap signal. |
| `centroid_distance` | `float` | Cosine distance to cluster centroid (or to the nearest centroid for noise points). |
| `is_outlier` | `bool` | `True` for HDBSCAN noise, or for KMeans samples above `mean + outlier_threshold × std`. |
| `umap_x`, `umap_y` | `float` | 2-D UMAP coordinates for visualization. |
| `embedding_error` | `str` | **v2** — empty when the embed succeeded; otherwise `"<ExceptionType>: <message>"` after 4 retries exhausted. Filter on this to find samples that failed (corrupt files, bad codecs, timeouts). |

## Operators

The plugin registers three operators plus one panel:

| Operator | What it does |
|----------|--------------|
| **Analyze Coverage** | The full pipeline: embed → cluster → describe (lazy by default) → detect gaps. Writes the sample fields above and `dataset.info["gap_report"]` + `dataset.info["coverage_history"]`. |
| **Show Gap Report** | Renders the three-tier report as typed FiftyOne views: `Header` + severity `Notice/Error/Warning/Success` + `TableView`s. The "Since last run" diff appears at the top when a prior run is on record. |
| **Export Coverage Report** | Writes a self-contained HTML file to a path you choose. Embeds the full three-tier report, an inline SVG UMAP scatter plot, and a base64-encoded CSV of per-cluster stats via a download link. No external assets — recipients without FiftyOne can open it in any browser. |
| **Coverage Map** (panel) | Interactive scatter: KDE heatmap backdrop, 4-bucket health-coloured clusters, red-diamond gap markers projected into UMAP space, on-click lazy Pegasus descriptions. |

## How It Works

```
Video Dataset
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. Embed    Marengo 3.0 → 512-d                                │
│              · SHA-256(first 10 MB) cache → re-runs ~instant    │
│              · 5× async concurrency · 3 retries w/ backoff      │
│              · samples that fail get `embedding_error`          │
└─────────┬───────────────────────────────────────────────────────┘
          ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. Cluster  HDBSCAN (cosine, eom) by default                   │
│              · noise → cluster_id=-1, is_outlier=True           │
│              · cluster_confidence from probabilities_           │
│              · fitted UMAP model cached; reused until drift     │
│                exceeds 30%, then retrained                      │
│              · num_clusters > 0 → KMeans fallback               │
└─────────┬───────────────────────────────────────────────────────┘
          ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. Describe (fast mode: lazy — no upfront Pegasus calls)        │
│              · full mode: 4 reps / cluster (2 nearest + 1       │
│                boundary + 1 random), parse COMMON / VARIATION   │
│              · cluster_label = joined COMMONs                   │
│              · cluster_diversity = deduped VARIATION phrases    │
└─────────┬───────────────────────────────────────────────────────┘
          ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. Detect Gaps                                                  │
│              · Hierarchical categories matched vs centroids     │
│                (parent + child level coverage)                  │
│              · priority_score ∈ [0, 100] per gap                │
│                (cosine distance + user-expected + isolation)    │
│              · Voronoi-based UMAP coverage score                │
│              · history entry appended for next run's diff        │
└─────────┬───────────────────────────────────────────────────────┘
          ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. Surface  · Coverage Map panel (scatter + heatmap + diamonds)│
│              · Show Gap Report (3-tier typed output)            │
│              · Export Coverage Report (self-contained HTML)     │
└─────────────────────────────────────────────────────────────────┘
```

### Cluster health colours

A 4-bucket palette used consistently across the scatter plot, the operator TableView, the markdown report, and the HTML export:

- 🟢 **well-covered** — 6+ samples · green (`#2e7d32`)
- 🟡 **thin** — 3–5 samples · amber (`#f9a825`)
- 🔴 **sparse** — under 3 samples · red (`#c62828`)
- ⚫ **noise** — HDBSCAN outliers · gray (`#8a8a8a`)

### Gap priority scoring

Every detected gap gets a `priority_score ∈ [0, 100]` blending three linearly-normalised factors with configurable weights:

| Factor | Meaning | Default weight |
|--------|---------|----------------|
| distance | Cosine distance from the nearest cluster — how far the gap sits from existing data | 0.5 |
| expected | 1 if the gap came from the user's `expected_categories` list, else 0 | 0.3 |
| isolation | Cosine distance between the nearest cluster and its nearest peer cluster — gaps in already-isolated regions are harder to fill | 0.2 |

Severity badges in the Priority Gaps table: **Critical** (> 70), **Moderate** (40–70), **Low** (< 40).

### Coverage history & diff

After every run, the plugin appends a lightweight entry to `dataset.info["coverage_history"]`:

```json
{
  "timestamp": "2026-04-18T12:34:56",
  "coverage_score": 0.42,
  "n_samples": 40,
  "n_clusters": 3,
  "gaps": [{"name": "elevated platform", "priority_score": 98.0, "type": "category"}, ...]
}
```

When a prior entry exists, subsequent reports show a "Since last run" block at the top with:

- Coverage delta (pct points)
- New-sample count
- Closed gaps (were missing, now covered)
- Still-open gaps (missing in both runs)
- Newly-opened gaps

## Example Output

**Tier 1 banner** (operator panel):

> **🟢 Coverage: 75.0% — Good**  · *Prioritize collecting videos that depict 'elevated platform' (nearest existing cluster: 'outdoor ladder climb') — priority 98/100, the most urgent gap.*

**Tier 2 · Priority Gaps** (TableView, top 3 of ranked list):

| # | Severity | Priority | Gap | Closest Cluster | Sim | Recommended action |
|---|----------|----------|-----|-----------------|-----|--------------------|
| 1 | Critical | 98/100 | elevated platform | outdoor ladder climb | 0.02 | Collect ~8 videos depicting 'elevated platform' from varied angles, subjects, and settings. |
| 2 | Critical | 94/100 | loading | warehouse forklift forward | 0.08 | Collect ~8 videos depicting 'loading' from varied angles, subjects, and settings. |
| 3 | Moderate | 55/100 | Sparse cluster 5: garage vehicle inspection | garage vehicle inspection | — | Collect ~5 more videos similar to 'garage vehicle inspection' (currently 2) to densify the cluster. |

**Since last run** (when history exists):

> Coverage: 35% → 42% (+7 pts) · Samples: 40 → 45 (+5)
> ✓ Closed (2): loading, turning
> ✗ Still open (1): elevated platform

## Troubleshooting

**Samples failed to embed** — any permanent Twelve Labs error (corrupt file, unsupported codec, network timeout) after 3 retries stamps a reason on `sample.embedding_error`. Find them with:

```python
ds.match(F("embedding_error") != "").count()
ds.match(F("embedding_error") != "").first().filepath
```

**Re-run after removing a cached embedding** — set `clear_cache=True` in the operator, or delete `~/.video_gap_analyzer/embeddings.db`.

**Force UMAP retraining** — delete `~/.video_gap_analyzer/umap_models/<dataset_name>/`.

**Pegasus descriptions don't appear** — with `use_pegasus=False` (the default), click a cluster point in the Coverage Map panel to trigger a single Pegasus call. The description caches for the rest of the session and persists to `cluster_label` / `cluster_diversity` on every sample in the cluster.

## Built With

- [FiftyOne](https://docs.voxel51.com/) — dataset management + interactive app
- [Twelve Labs Marengo 3.0](https://docs.twelvelabs.io/) — multimodal video + text embeddings
- [Twelve Labs Pegasus 1.2](https://docs.twelvelabs.io/) — video-to-text generation
- [HDBSCAN](https://hdbscan.readthedocs.io/) — density-based clustering (default)
- [scikit-learn](https://scikit-learn.org/) — KMeans fallback + distance metrics
- [UMAP](https://umap-learn.readthedocs.io/) — 2-D projection (optionally parametric via TensorFlow)
- [SciPy](https://scipy.org/) — Voronoi coverage + gaussian_kde heatmap
- [NumPy](https://numpy.org/) — vector arithmetic

## Changelog — v1 → v2

Three months of hardening after the hackathon baseline. Every change below lives on the `v2-improvements` branch.

### Phase 1 · Embedding performance

- Persistent **SQLite embedding cache** at `~/.video_gap_analyzer/embeddings.db`, keyed by SHA-256 of the first 10 MB of each video. Re-runs short-circuit the Marengo API entirely for known-content videos.
- **Async embedding pipeline** — 5× concurrency via `AsyncTwelveLabs` + `asyncio.Semaphore(5)`. Progress is driven by `asyncio.as_completed` so the bar advances correctly under concurrency.
- **`clear_cache` operator input** — wipes both the SQLite cache and every sample's `embedding` field so the next run fires fresh API calls for everything.
- Operator panel now reports `n_fresh_embeddings / n_from_cache / n_already_embedded / n_failed_embeddings` separately so the cache benefit is visible every run.

### Phase 2 · Clustering quality

- **HDBSCAN** replaces KMeans as the default (metric=cosine, cluster_selection_method=eom, min_cluster_size=3). Silhouette auto-k loop is gone. `num_clusters=0` → HDBSCAN; `> 0` → KMeans fallback.
- **Noise handling** — `cluster_id = -1` + `is_outlier = True` for HDBSCAN noise points. The Coverage Map panel renders them as gray hollow circles, visually distinct from clustered points.
- **`cluster_confidence` field** — HDBSCAN's `probabilities_` vector stored per-sample. Low-but-nonzero values mark cluster-boundary points as a soft gap signal.
- **Operator UI** exposes both paths: "Number of Clusters (0 = HDBSCAN auto)" with a description that spells out the fallback behaviour.

### Phase 3 · Smarter gap detection

- **Hierarchical categories** — `"parent: child1, child2 | parent2: ..."` format parsed into a tree. Each leaf embeds via Marengo text and matches against **cluster centroids** (more stable than per-sample matches). Reports show per-parent coverage + per-child status.
- **Gap priority score** — every gap gets a `priority_score ∈ [0, 100]` blending cosine distance, user-expected flag, and cluster isolation. Configurable weights; default 0.5 / 0.3 / 0.2.
- **Voronoi-based coverage score** — replaces the 10×10 grid heuristic. Builds a Voronoi tessellation over UMAP coords, clips each cell to the convex hull, and returns the hull-area fraction occupied by cells below `median × 1.5`.
- **Pegasus sampling** upgraded — 4 diverse reps per cluster (2 nearest-centroid + 1 boundary + 1 random, seeded for reproducibility). Prompt returns labeled `COMMON:` and `VARIATION:` sections; results land in `cluster_label` + `cluster_diversity`.

### Phase 4 · Tiered report system

- **New `gap_report.py` module** — pure builders for three tiers (Executive Summary, Priority Gaps, Full Breakdown) plus a markdown renderer.
- **`ShowGapReport` typed output** — Header, severity-coded Notice/Error/Warning/Success (red for Poor coverage, yellow for Fair, green for Good/Excellent), and TableViews replace the old markdown dump.
- **New `ExportCoverageReport` operator** — writes a single-file HTML report with inline SVG scatter, base64 CSV download, and the full three-tier narrative.
- **Coverage history** — `dataset.info["coverage_history"]` gets an append-only entry after each run. Subsequent reports show a **"Since last run"** block at the top (coverage delta, new samples, closed / still-open / newly-opened gaps), severity-coded by trend.

### Phase 5 · Visualization upgrades

- **KDE density heatmap** as the bottom layer of the Coverage Map — warm colours mark dense regions, transparent at zero so gaps are visually obvious.
- **Expected-category markers** projected into UMAP space via `umap_reducer.transform(text_embedding)` — red hollow diamonds with visible labels land in the actual sparse zones of the heatmap.
- **Lazy Pegasus** — `use_pegasus` default flipped to `False`. Click any cluster point to fire a single Pegasus call on its nearest-centroid rep; result caches for the session and persists to sample fields so the next run starts warm.
- **4-bucket cluster health colour scheme** (🟢 well-covered / 🟡 thin / 🔴 sparse / ⚫ noise) applied consistently across scatter plot, operator TableView, markdown report, and HTML export. Phase 4.3's HTML export gained `.status-{bucket}` CSS rules + a colour-swatch legend.

### Phase 6 · Performance & resilience

- **UMAP model cache** at `~/.video_gap_analyzer/umap_models/<dataset_name>/`. First run fits and saves; subsequent runs load and `transform()`. Change detection: if more than 30% of samples are new since the last training, retrain from scratch; otherwise just transform the new points. Supports parametric UMAP (TensorFlow) with transparent pickle fallback for regular `umap.UMAP`.
- **Unified retry logic** — every Twelve Labs call (Marengo embed, Marengo text, Pegasus analyze, asset upload) runs through `_retry_sync` or `_retry_async`: 3 retries, 2s → 4s → 8s exponential backoff.
- **Per-sample `embedding_error`** — permanent failures (retries exhausted) write `"<ExceptionType>: <message>"` onto the sample so users can filter. Stage 1 aggregates a `{reason: count}` map and the operator progress label calls out the skip count + top reason.

### Notebooks

One self-contained Jupyter notebook per phase, driven by the plugin's own production modules via a synthetic-package loader:

- `notebooks/05_embedding_cache_demo.ipynb` — Phase 1.1 cache internals
- `notebooks/06_hdbscan_clustering_demo.ipynb` — Phase 2 clustering quality
- `notebooks/07_smarter_gap_detection_demo.ipynb` — Phase 3 hierarchy + priority + Voronoi + diverse Pegasus
- `notebooks/08_tiered_report_demo.ipynb` — Phase 4 tiered report + HTML export + history diff
- `notebooks/09_visualization_upgrades_demo.ipynb` — Phase 5 KDE + projected markers + health palette
- `notebooks/10_performance_and_resilience_demo.ipynb` — Phase 6 UMAP caching + retry backoff + embedding_error

## Hackathon Context

Built at the **Video Understanding AI Hackathon** at Northeastern University, April 3, 2026. v2 hardening done in the weeks following. Inspired by the [CV4Smalls](https://cv4smalls.netlify.app/) workshop series and the insight that understanding your data distribution matters more than throwing bigger models at incomplete datasets.
