# Video Content Gap Analyzer

**Find what's missing in your video dataset before you waste time collecting more of what you already have.**

A [FiftyOne](https://docs.voxel51.com/) plugin that identifies coverage gaps in video datasets using [Twelve Labs](https://twelvelabs.io/) Marengo embeddings, KMeans clustering, and Pegasus descriptions.

## The Problem

Most teams improve their video models by collecting *more* data. But more data doesn't help if it's more of the same. The [CV4Smalls](https://cv4smalls.netlify.app/) workshop series demonstrated that **data curation beats model complexity** — the winning EAR 2025 submission used a model architecture from 2019 and won by picking better training data. Before you collect another 10,000 videos, you should know what your dataset is actually missing.

## What This Plugin Does

1. **Embeds** your video dataset into 512-d vectors using Twelve Labs Marengo 3.0
2. **Clusters** videos by semantic similarity with auto-tuned KMeans
3. **Describes** each cluster in plain English via Twelve Labs Pegasus 1.2
4. **Detects gaps** — sparse clusters, isolated clusters, missing expected categories
5. **Visualizes** everything in an interactive scatter plot inside the FiftyOne App

## Quick Start

### Install

```bash
# Clone and register as a FiftyOne plugin
git clone https://github.com/rishimule/video-content-gap-analyzer.git
fiftyone plugins create video-content-gap-analyzer --from-dir ./video-content-gap-analyzer

# Or install directly from GitHub
fiftyone plugins download https://github.com/rishimule/video-content-gap-analyzer

# Install dependencies
pip install -r video-content-gap-analyzer/requirements.txt
```

### Set your API key

```bash
export TWELVELABS_API_KEY="your_key_here"
```

### Run

```python
import fiftyone as fo
import fiftyone.zoo as foz

# Load any video dataset
dataset = foz.load_zoo_dataset("quickstart-video", max_samples=20)

# Launch the App and run "Analyze Coverage" from the operator menu
session = fo.launch_app(dataset)
```

Open the operator browser (press `` ` `` in the App), search for **Analyze Coverage**, configure your parameters, and run. Results appear as sample fields and in the **Coverage Panel**.

## How It Works

```
Video Dataset
    │
    ▼
┌─────────────────────┐
│  1. Embed           │  Twelve Labs Marengo 3.0 → 512-d vectors per video
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  2. Cluster         │  KMeans (auto-k via silhouette score) + UMAP 2D projection
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  3. Describe        │  Pegasus 1.2 generates plain-English label per cluster
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  4. Detect Gaps     │  Sparse clusters, isolated clusters, missing categories
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  5. Visualize       │  Interactive scatter plot + gap summary in FiftyOne App
└─────────────────────┘
```

**Embedding** — Each video is uploaded to Twelve Labs and embedded using Marengo 3.0 (visual + audio fusion). Already-embedded samples are skipped on re-runs.

**Clustering** — L2-normalized embeddings are clustered with KMeans. Set `num_clusters=0` to auto-select *k* via silhouette scoring. Outliers are flagged when their centroid distance exceeds mean + 2σ. UMAP reduces to 2D for visualization.

**Describing** — The two samples closest to each cluster centroid are sent to Pegasus 1.2, which generates a one-sentence description. These become human-readable cluster labels.

**Gap Detection** produces three signals:
- **Sparse clusters** — clusters with fewer than 3 samples (underrepresented content)
- **Isolated clusters** — clusters whose centroids are far from all others (unusual content)
- **Category gaps** — if you provide expected categories (e.g., "person falling, forklift moving"), the plugin embeds them with Marengo and measures cosine similarity against your data. Low-similarity categories are flagged as gaps.

**Coverage score** — A 0–1 score based on how much of the UMAP embedding space your dataset actually occupies (10×10 grid occupancy).

## Example Output

```
Gap Report
──────────────────────────────────
Coverage Score: 0.42

Sparse Clusters (underrepresented):
  • Cluster 3: "Person climbing ladder in warehouse" — 2 samples
  • Cluster 7: "Emergency vehicle arriving at scene" — 1 sample

Isolated Clusters (unusual content):
  • Cluster 7: "Emergency vehicle arriving at scene" — mean distance 0.8231

Category Gaps (missing from dataset):
  • "forklift moving"  — best match: "Person walking in warehouse" (sim: 0.18)
  • "fire evacuation"  — best match: "Emergency vehicle arriving" (sim: 0.24)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_clusters` | `5` | Number of KMeans clusters. Set to `0` for auto-selection via silhouette score. |
| `expected_categories` | `""` | Comma-separated categories to check for (e.g., `person falling, forklift moving`). |
| `use_pegasus` | `True` | Generate natural language cluster labels with Pegasus. Disable for faster runs. |
| `max_samples` | `0` | Max samples to process (`0` = all). Useful for limiting API cost on large datasets. |
| `outlier_threshold` | `2.0` | Std deviations above mean centroid distance to flag outliers. Lower = more outliers. |
| `gap_threshold` | `0.3` | Cosine similarity below which a category is flagged as missing. Higher = more gaps. |

## Fields Written to Samples

| Field | Type | Description |
|-------|------|-------------|
| `embedding` | `list[float]` | 512-d Marengo video embedding |
| `cluster_id` | `int` | KMeans cluster assignment |
| `cluster_label` | `str` | Pegasus-generated cluster description |
| `centroid_distance` | `float` | Cosine distance to cluster centroid |
| `is_outlier` | `bool` | Whether distance exceeds threshold |
| `umap_x`, `umap_y` | `float` | 2D UMAP coordinates for visualization |

## Built With

- [FiftyOne](https://docs.voxel51.com/) — Dataset management and visualization
- [Twelve Labs Marengo 3.0](https://docs.twelvelabs.io/) — Multimodal video/text embeddings
- [Twelve Labs Pegasus 1.2](https://docs.twelvelabs.io/) — Video-to-text generation
- [scikit-learn](https://scikit-learn.org/) — KMeans clustering and silhouette scoring
- [UMAP](https://umap-learn.readthedocs.io/) — Dimensionality reduction
- [NumPy](https://numpy.org/) — Distance calculations

## Hackathon Context

Built at the **Video Understanding AI Hackathon** at Northeastern University, April 3, 2026. Inspired by the [CV4Smalls](https://cv4smalls.netlify.app/) workshop series and the insight that understanding your data distribution matters more than throwing bigger models at incomplete datasets.
