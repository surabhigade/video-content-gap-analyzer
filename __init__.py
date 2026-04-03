"""
Video Content Gap Analyzer — FiftyOne Plugin

Identifies coverage gaps in video datasets using Twelve Labs Marengo
embeddings and Pegasus descriptions. Two operators:

  - analyze_coverage: Full pipeline (embed -> cluster -> describe -> detect gaps)
  - show_gap_report: Display the gap report from the last analysis run
"""

import os
import time
import logging
from datetime import datetime
from collections import defaultdict

import numpy as np
import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.preprocessing import normalize
import umap

from twelvelabs import TwelveLabs, VideoInputRequest, MediaSource, TextInputRequest
from twelvelabs.types import VideoContext_AssetId
from twelvelabs.indexes import IndexesCreateRequestModelsItem

logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================

# Clustering
KMEANS_N_INIT = 10
KMEANS_RANDOM_STATE = 42
OUTLIER_STD_FACTOR = 2.0

# UMAP
UMAP_N_COMPONENTS = 2
UMAP_METRIC = "cosine"
UMAP_MIN_DIST = 0.1
UMAP_RANDOM_STATE = 42

# Pegasus descriptions
REPS_PER_CLUSTER = 2
INDEX_NAME_PREFIX = "gap-analyzer"
PEGASUS_PROMPT = (
    "Describe the main activity, setting, and key objects visible "
    "in this video in one concise sentence."
)
POLL_INTERVAL = 10.0
RATE_LIMIT_WAIT = 30

# Gap detection
SPARSE_THRESHOLD = 3
GAP_SIMILARITY_THRESHOLD = 0.10
ISOLATION_STD_FACTOR = 1.5
COVERAGE_GRID_SIZE = 10


# ============================================================
# Helper functions — API setup
# ============================================================

def get_twelvelabs_client():
    """Validate API key and return TwelveLabs client."""
    api_key = os.environ.get("TWELVELABS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TWELVELABS_API_KEY environment variable is not set. "
            "Set it with: export TWELVELABS_API_KEY=<your-key>"
        )
    return TwelveLabs(api_key=api_key)


# ============================================================
# Helper functions — Embedding (from notebook 01)
# ============================================================

def embed_sample(client, sample):
    """Embed a single video sample via Marengo 3.0.

    Returns True on success, False on failure, None if skipped.
    """
    filepath = sample.filepath
    filename = os.path.basename(filepath)

    # Skip if already embedded
    try:
        if sample["embedding"] is not None:
            return None
    except (KeyError, AttributeError):
        pass

    try:
        with open(filepath, "rb") as f:
            asset = client.assets.create(method="direct", file=f)

        response = client.embed.v_2.create(
            input_type="video",
            model_name="marengo3.0",
            video=VideoInputRequest(
                media_source=MediaSource(asset_id=asset.id),
                embedding_option=["visual", "audio"],
                embedding_scope=["asset"],
                embedding_type=["fused_embedding"],
            ),
        )

        embedding = response.data[0].embedding
        sample["embedding"] = embedding
        sample.save()
        return True

    except Exception as e:
        logger.warning("Failed to embed %s: %s", filename, e)
        return False


def embed_all_samples(client, dataset, ctx):
    """Embed all samples in dataset, reporting progress in the 0.0-0.25 range.

    Returns (success_count, fail_count, skip_count).
    """
    total = len(dataset)
    success_count = 0
    fail_count = 0
    skip_count = 0

    for i, sample in enumerate(dataset):
        filename = os.path.basename(sample.filepath)
        ctx.set_progress(
            progress=0.25 * (i / max(total, 1)),
            label=f"Embedding video {i + 1}/{total}: {filename}",
        )

        result = embed_sample(client, sample)
        if result is None:
            skip_count += 1
        elif result:
            success_count += 1
        else:
            fail_count += 1

    dataset.save()
    return success_count, fail_count, skip_count


# ============================================================
# Helper functions — Clustering (from notebook 02)
# ============================================================

def run_clustering(dataset, n_clusters):
    """Run KMeans clustering, outlier detection, and UMAP on embedded samples.

    Writes cluster_id, centroid_distance, is_outlier, umap_x, umap_y to each
    sample. Returns the number of samples processed.
    """
    # Extract embedded samples
    samples_list = []
    embeddings_list = []

    for sample in dataset:
        try:
            emb = sample["embedding"]
        except (KeyError, AttributeError):
            continue
        if emb is not None:
            samples_list.append(sample)
            embeddings_list.append(emb)

    n_samples = len(embeddings_list)
    if n_samples == 0:
        raise RuntimeError(
            "No samples have embeddings. Embedding stage may have failed."
        )

    embeddings = np.array(embeddings_list)
    embeddings_norm = normalize(embeddings, norm="l2")

    # Clamp n_clusters to valid range
    if n_clusters > n_samples:
        logger.warning(
            "n_clusters (%d) > n_samples (%d), clamping", n_clusters, n_samples
        )
        n_clusters = max(1, n_samples - 1) if n_samples > 1 else 1

    # KMeans
    if n_clusters <= 1:
        labels = np.zeros(n_samples, dtype=int)
        centroids_norm = embeddings_norm.mean(axis=0, keepdims=True)
        centroids_norm = normalize(centroids_norm, norm="l2")
    else:
        kmeans = KMeans(
            n_clusters=n_clusters, random_state=KMEANS_RANDOM_STATE,
            n_init=KMEANS_N_INIT,
        )
        labels = kmeans.fit_predict(embeddings_norm)
        centroids_norm = normalize(kmeans.cluster_centers_, norm="l2")

    # Centroid distances
    distances = np.array([
        cosine_distances(
            embeddings_norm[i:i + 1],
            centroids_norm[labels[i]:labels[i] + 1],
        )[0, 0]
        for i in range(n_samples)
    ])

    mean_dist = distances.mean()
    std_dist = distances.std()

    # Outlier detection
    if std_dist > 0:
        threshold = mean_dist + OUTLIER_STD_FACTOR * std_dist
        is_outlier = distances > threshold
    else:
        is_outlier = np.zeros(n_samples, dtype=bool)

    # UMAP 2D reduction
    if n_samples < 2:
        coords_2d = np.zeros((n_samples, 2))
    else:
        n_neighbors = min(5, n_samples - 1)
        reducer = umap.UMAP(
            n_components=UMAP_N_COMPONENTS,
            metric=UMAP_METRIC,
            n_neighbors=n_neighbors,
            min_dist=UMAP_MIN_DIST,
            random_state=UMAP_RANDOM_STATE,
        )
        coords_2d = reducer.fit_transform(embeddings_norm)

    # Write fields to samples
    for i, sample in enumerate(samples_list):
        sample["cluster_id"] = int(labels[i])
        sample["centroid_distance"] = float(distances[i])
        sample["is_outlier"] = bool(is_outlier[i])
        sample["umap_x"] = float(coords_2d[i, 0])
        sample["umap_y"] = float(coords_2d[i, 1])
        sample.save()

    dataset.save()
    return n_samples


# ============================================================
# Helper functions — Pegasus descriptions (from notebook 03)
# ============================================================

def find_cluster_representatives(dataset):
    """For each cluster_id, find the REPS_PER_CLUSTER samples closest to centroid."""
    clusters = defaultdict(list)

    for sample in dataset:
        try:
            cid = sample["cluster_id"]
            dist = sample["centroid_distance"]
        except (KeyError, AttributeError):
            continue
        if cid is not None and dist is not None:
            clusters[cid].append((dist, sample))

    representatives = {}
    for cid in sorted(clusters.keys()):
        sorted_samples = sorted(clusters[cid], key=lambda x: x[0])
        representatives[cid] = [s for _, s in sorted_samples[:REPS_PER_CLUSTER]]

    return representatives


def upload_asset(client, filepath):
    """Upload a video file as a Twelve Labs asset. Returns asset or None."""
    try:
        with open(filepath, "rb") as f:
            asset = client.assets.create(method="direct", file=f)
        return asset
    except Exception as e:
        logger.warning("Upload failed for %s: %s", os.path.basename(filepath), e)
        return None


def analyze_via_asset(client, asset_id, prompt):
    """Approach A: analyze directly using asset_id (no indexing)."""
    response = client.analyze(
        video=VideoContext_AssetId(asset_id=asset_id),
        prompt=prompt,
        temperature=0.2,
    )
    return response.data


def create_pegasus_index(client):
    """Create a Twelve Labs index with pegasus1.2 model support."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    index_name = f"{INDEX_NAME_PREFIX}-{timestamp}"

    index = client.indexes.create(
        index_name=index_name,
        models=[
            IndexesCreateRequestModelsItem(
                model_name="pegasus1.2",
                model_options=["visual", "audio"],
            ),
        ],
    )
    logger.info("Created Pegasus index: %s", index.id)
    return index.id


def index_and_analyze(client, index_id, asset_id, prompt):
    """Approach B: index the asset, wait for ready, then analyze."""
    resp = client.indexes.indexed_assets.create(
        index_id=index_id,
        asset_id=asset_id,
    )
    indexed_asset_id = resp.id

    # Poll until ready
    while True:
        detail = client.indexes.indexed_assets.retrieve(index_id, indexed_asset_id)
        if detail.status == "ready":
            break
        if detail.status == "failed":
            raise RuntimeError(f"Indexing failed for asset {asset_id}")
        time.sleep(POLL_INTERVAL)

    response = client.analyze(
        video_id=indexed_asset_id,
        prompt=prompt,
        temperature=0.2,
    )
    return response.data


def generate_description(client, filepath, use_indexing, index_id):
    """Generate a Pegasus description for a video.

    Returns (description_or_None, use_indexing_updated).
    Handles rate limits and approach fallback (A -> B).
    """
    asset = upload_asset(client, filepath)
    if asset is None:
        return None, use_indexing

    for attempt in range(2):
        try:
            if use_indexing:
                text = index_and_analyze(client, index_id, asset.id, PEGASUS_PROMPT)
            else:
                text = analyze_via_asset(client, asset.id, PEGASUS_PROMPT)

            if text:
                return text.strip(), use_indexing
            else:
                return None, use_indexing

        except Exception as e:
            error_str = str(e).lower()

            # Rate limit: wait and retry once
            if "429" in str(e) or "rate" in error_str or "too many" in error_str:
                if attempt == 0:
                    logger.info("Rate limited, waiting %ds...", RATE_LIMIT_WAIT)
                    time.sleep(RATE_LIMIT_WAIT)
                    continue
                else:
                    return None, use_indexing

            # Non-rate-limit error on Approach A: switch to B
            if not use_indexing:
                logger.info("Direct analysis failed, switching to index-based approach")
                return None, True

            # Approach B failure
            logger.warning("Description generation failed: %s", e)
            return None, use_indexing

    return None, use_indexing


def generate_cluster_labels(client, dataset, use_pegasus, ctx):
    """Generate labels for each cluster and write cluster_label to all samples.

    Returns dict {cluster_id: label_str}.
    """
    representatives = find_cluster_representatives(dataset)
    cluster_labels = {}
    n_clusters = len(representatives)

    if not use_pegasus:
        # Fast path: generic labels
        for cid in sorted(representatives.keys()):
            cluster_labels[cid] = f"Cluster {cid}"
    else:
        use_indexing = False
        index_id = None

        for idx, cid in enumerate(sorted(representatives.keys())):
            ctx.set_progress(
                progress=0.50 + 0.25 * (idx / max(n_clusters, 1)),
                label=f"Describing cluster {cid + 1}/{n_clusters}...",
            )

            samples = representatives[cid]
            descriptions = []

            if use_indexing and index_id is None:
                index_id = create_pegasus_index(client)

            for sample in samples:
                desc, use_indexing_new = generate_description(
                    client, sample.filepath, use_indexing, index_id
                )

                # Handle approach switch
                if use_indexing_new and not use_indexing:
                    use_indexing = True
                    if index_id is None:
                        index_id = create_pegasus_index(client)
                    desc, _ = generate_description(
                        client, sample.filepath, use_indexing, index_id
                    )

                if desc:
                    descriptions.append(desc)

            # Combine descriptions into label
            if len(descriptions) >= 2:
                cluster_labels[cid] = f"{descriptions[0]}; {descriptions[1]}"
            elif len(descriptions) == 1:
                cluster_labels[cid] = descriptions[0]
            else:
                cluster_labels[cid] = f"Cluster {cid}"

    # Apply labels to all samples
    for sample in dataset:
        try:
            cid = sample["cluster_id"]
        except (KeyError, AttributeError):
            continue
        if cid is not None and cid in cluster_labels:
            sample["cluster_label"] = cluster_labels[cid]
            sample.save()

    dataset.save()
    return cluster_labels


# ============================================================
# Helper functions — Gap detection (from notebook 04)
# ============================================================

def extract_cluster_data(dataset):
    """Extract embedded + clustered samples from the dataset.

    Returns (samples_list, embeddings, cluster_ids, umap_coords, cluster_labels_map).
    """
    samples_list = []
    embeddings_list = []
    cluster_ids_list = []
    umap_list = []
    cluster_labels_map = {}

    for sample in dataset:
        try:
            emb = sample["embedding"]
            cid = sample["cluster_id"]
            ux = sample["umap_x"]
            uy = sample["umap_y"]
        except (KeyError, AttributeError):
            continue

        if emb is None or cid is None or ux is None or uy is None:
            continue

        samples_list.append(sample)
        embeddings_list.append(emb)
        cluster_ids_list.append(cid)
        umap_list.append([ux, uy])

        if cid not in cluster_labels_map:
            try:
                label = sample["cluster_label"]
                cluster_labels_map[cid] = label if label else f"Cluster {cid}"
            except (KeyError, AttributeError):
                cluster_labels_map[cid] = f"Cluster {cid}"

    if len(samples_list) == 0:
        raise RuntimeError(
            "No samples have all required fields "
            "(embedding, cluster_id, umap_x, umap_y)."
        )

    embeddings = np.array(embeddings_list)
    cluster_ids = np.array(cluster_ids_list, dtype=int)
    umap_coords = np.array(umap_list)

    return samples_list, embeddings, cluster_ids, umap_coords, cluster_labels_map


def compute_centroids(embeddings_norm, cluster_ids):
    """Recompute L2-normalized cluster centroids from sample embeddings."""
    unique_ids = np.sort(np.unique(cluster_ids))
    centroids = []

    for cid in unique_ids:
        mask = cluster_ids == cid
        centroid = embeddings_norm[mask].mean(axis=0)
        centroids.append(centroid)

    centroids = np.array(centroids)
    centroids = normalize(centroids, norm="l2")
    return centroids, unique_ids


def detect_sparse_clusters(cluster_ids, cluster_labels_map, threshold=SPARSE_THRESHOLD):
    """Flag clusters with fewer than threshold samples."""
    unique, counts = np.unique(cluster_ids, return_counts=True)
    sparse = []

    for cid, cnt in zip(unique, counts):
        if cnt < threshold:
            sparse.append({
                "cluster_id": int(cid),
                "label": cluster_labels_map.get(int(cid), f"Cluster {cid}"),
                "count": int(cnt),
            })

    return sparse


def detect_isolated_clusters(centroids, unique_ids, cluster_labels_map):
    """Identify clusters whose centroids are far from all others."""
    n_clusters = len(unique_ids)
    if n_clusters < 3:
        return []

    dist_matrix = cosine_distances(centroids, centroids)

    mean_dists = []
    for i in range(n_clusters):
        others = [dist_matrix[i, j] for j in range(n_clusters) if j != i]
        mean_dists.append(np.mean(others))

    mean_dists = np.array(mean_dists)
    threshold = mean_dists.mean() + ISOLATION_STD_FACTOR * mean_dists.std()

    isolated = []
    for i, cid in enumerate(unique_ids):
        if mean_dists[i] > threshold:
            isolated.append({
                "cluster_id": int(cid),
                "label": cluster_labels_map.get(int(cid), f"Cluster {cid}"),
                "mean_inter_distance": round(float(mean_dists[i]), 4),
            })

    return isolated


def compute_umap_coverage(umap_coords, grid_size=COVERAGE_GRID_SIZE):
    """Compute what fraction of the UMAP bounding box is occupied."""
    x_min, x_max = umap_coords[:, 0].min(), umap_coords[:, 0].max()
    y_min, y_max = umap_coords[:, 1].min(), umap_coords[:, 1].max()

    x_range = x_max - x_min
    y_range = y_max - y_min

    if x_range == 0 or y_range == 0:
        return 0.0

    x_min -= 0.01 * x_range
    x_max += 0.01 * x_range
    y_min -= 0.01 * y_range
    y_max += 0.01 * y_range

    cell_w = (x_max - x_min) / grid_size
    cell_h = (y_max - y_min) / grid_size

    occupied = set()
    for x, y in umap_coords:
        cx = min(int((x - x_min) / cell_w), grid_size - 1)
        cy = min(int((y - y_min) / cell_h), grid_size - 1)
        occupied.add((cx, cy))

    return len(occupied) / (grid_size * grid_size)


def embed_categories(client, categories):
    """Embed category strings via Twelve Labs Marengo text API.

    Returns dict {category_str: np.ndarray(512,)} for successful embeddings.
    """
    results = {}

    for i, category in enumerate(categories, start=1):
        for attempt in range(2):
            try:
                response = client.embed.v_2.create(
                    input_type="text",
                    model_name="marengo3.0",
                    text=TextInputRequest(input_text=category),
                )

                embedding = np.array(response.data[0].embedding)
                embedding = normalize(embedding.reshape(1, -1), norm="l2")[0]
                results[category] = embedding
                break

            except Exception as e:
                error_str = str(e).lower()

                if "429" in str(e) or "rate" in error_str or "too many" in error_str:
                    if attempt == 0:
                        logger.info("Rate limited on category embed, waiting %ds...", RATE_LIMIT_WAIT)
                        time.sleep(RATE_LIMIT_WAIT)
                        continue

                logger.warning("Failed to embed category '%s': %s", category, e)
                break

    return results


def detect_category_gaps(category_embeddings, embeddings_norm, cluster_ids,
                         unique_ids, cluster_labels_map,
                         threshold=GAP_SIMILARITY_THRESHOLD):
    """Compare each category embedding to all sample embeddings."""
    categories = list(category_embeddings.keys())
    cat_matrix = np.array([category_embeddings[c] for c in categories])

    sim_matrix = cosine_similarity(cat_matrix, embeddings_norm)

    results = []
    for i, category in enumerate(categories):
        max_sim = float(sim_matrix[i].max())
        closest_sample_idx = int(sim_matrix[i].argmax())
        closest_cid = int(cluster_ids[closest_sample_idx])
        closest_label = cluster_labels_map.get(closest_cid, f"Cluster {closest_cid}")

        results.append({
            "category": category,
            "closest_cluster": closest_label,
            "closest_cluster_id": closest_cid,
            "similarity": round(max_sim, 4),
            "is_gap": max_sim < threshold,
        })

    return results


def tag_sparse_samples(dataset, sparse_cluster_ids):
    """Tag samples in sparse clusters with 'sparse_cluster'."""
    tagged = 0
    for sample in dataset:
        try:
            cid = sample["cluster_id"]
        except (KeyError, AttributeError):
            continue
        if cid is not None and cid in sparse_cluster_ids:
            if "sparse_cluster" not in sample.tags:
                sample.tags.append("sparse_cluster")
                sample.save()
                tagged += 1
    return tagged


def detect_gaps(client, dataset, expected_categories, ctx):
    """Run full gap detection (structural + category-driven).

    Returns gap_report dict.
    """
    ctx.set_progress(progress=0.75, label="Extracting cluster data...")

    samples_list, embeddings, cluster_ids, umap_coords, cluster_labels_map = \
        extract_cluster_data(dataset)

    embeddings_norm = normalize(embeddings, norm="l2")
    centroids, unique_ids = compute_centroids(embeddings_norm, cluster_ids)

    # Structural analysis
    ctx.set_progress(progress=0.80, label="Detecting sparse and isolated clusters...")
    sparse_clusters = detect_sparse_clusters(cluster_ids, cluster_labels_map)
    isolated_clusters = detect_isolated_clusters(centroids, unique_ids, cluster_labels_map)
    umap_coverage = compute_umap_coverage(umap_coords)

    # Category-driven analysis
    category_results = []
    category_coverage = 0.0

    if expected_categories:
        ctx.set_progress(progress=0.85, label="Embedding expected categories...")
        category_embeddings = embed_categories(client, expected_categories)

        if category_embeddings:
            ctx.set_progress(progress=0.90, label="Computing category gaps...")
            category_results = detect_category_gaps(
                category_embeddings, embeddings_norm, cluster_ids,
                unique_ids, cluster_labels_map,
            )
            n_covered = sum(1 for cr in category_results if not cr["is_gap"])
            category_coverage = n_covered / len(category_results)

    # Tag sparse samples
    ctx.set_progress(progress=0.95, label="Tagging sparse cluster samples...")
    sparse_ids = {sc["cluster_id"] for sc in sparse_clusters}
    tag_sparse_samples(dataset, sparse_ids)

    # Compute coverage score
    if category_results:
        combined_score = 0.5 * umap_coverage + 0.5 * category_coverage
    else:
        combined_score = umap_coverage

    gap_report = {
        "sparse_clusters": sparse_clusters,
        "category_gaps": [
            {
                "category": cr["category"],
                "closest_cluster": cr["closest_cluster"],
                "similarity": cr["similarity"],
            }
            for cr in category_results if cr["is_gap"]
        ],
        "coverage_score": round(float(combined_score), 4),
    }

    return gap_report


# ============================================================
# Operator 1: AnalyzeCoverage
# ============================================================

class AnalyzeCoverage(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="analyze_coverage",
            label="Analyze Coverage",
            description=(
                "Run the full video content gap analysis pipeline: "
                "embed videos, cluster, generate descriptions, and detect gaps."
            ),
            allow_immediate_execution=True,
            allow_delegated_execution=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        inputs.int(
            "num_clusters",
            label="Number of Clusters",
            description="Number of clusters for grouping videos",
            default=5,
        )

        inputs.str(
            "expected_categories",
            label="Expected Categories",
            description=(
                "Comma-separated list of expected video categories to check "
                "against, e.g.: person falling, forklift moving, emergency evacuation"
            ),
            default="",
        )

        inputs.bool(
            "use_pegasus",
            label="Generate Cluster Descriptions (Pegasus)",
            description=(
                "Generate natural language cluster descriptions using Pegasus. "
                "If disabled, clusters get generic labels (faster)."
            ),
            default=True,
        )

        return types.Property(
            inputs,
            view=types.View(label="Video Content Gap Analyzer"),
        )

    def execute(self, ctx):
        dataset = ctx.dataset
        num_clusters = ctx.params.get("num_clusters", 5)
        expected_categories_str = ctx.params.get("expected_categories", "")
        use_pegasus = ctx.params.get("use_pegasus", True)

        # Parse categories
        expected_categories = [
            c.strip() for c in expected_categories_str.split(",") if c.strip()
        ] if expected_categories_str else []

        # Validate API key
        client = get_twelvelabs_client()

        # Stage 1: Embeddings (0.00 - 0.25)
        ctx.set_progress(progress=0.0, label="Stage 1/4: Generating video embeddings...")
        success, fail, skip = embed_all_samples(client, dataset, ctx)
        logger.info("Embeddings: %d new, %d failed, %d skipped", success, fail, skip)

        # Stage 2: Clustering (0.25 - 0.50)
        ctx.set_progress(progress=0.25, label="Stage 2/4: Clustering embeddings...")
        n_samples = run_clustering(dataset, num_clusters)
        ctx.set_progress(
            progress=0.50,
            label=f"Clustered {n_samples} samples into {num_clusters} groups",
        )

        # Stage 3: Cluster descriptions (0.50 - 0.75)
        ctx.set_progress(progress=0.50, label="Stage 3/4: Generating cluster descriptions...")
        cluster_labels = generate_cluster_labels(client, dataset, use_pegasus, ctx)

        # Stage 4: Gap detection (0.75 - 1.00)
        ctx.set_progress(progress=0.75, label="Stage 4/4: Detecting coverage gaps...")
        gap_report = detect_gaps(client, dataset, expected_categories, ctx)

        # Store report
        dataset.info["gap_report"] = gap_report
        dataset.save()

        ctx.set_progress(progress=1.0, label="Analysis complete!")

        return {
            "coverage_score": gap_report["coverage_score"],
            "n_sparse_clusters": len(gap_report["sparse_clusters"]),
            "n_category_gaps": len(gap_report["category_gaps"]),
            "n_samples": n_samples,
            "n_clusters": num_clusters,
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.float("coverage_score", label="Coverage Score (0-1)")
        outputs.int("n_sparse_clusters", label="Sparse Clusters Found")
        outputs.int("n_category_gaps", label="Category Gaps Found")
        outputs.int("n_samples", label="Samples Analyzed")
        outputs.int("n_clusters", label="Clusters Created")
        return types.Property(
            outputs,
            view=types.View(label="Coverage Analysis Complete"),
        )


# ============================================================
# Operator 2: ShowGapReport
# ============================================================

class ShowGapReport(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="show_gap_report",
            label="Show Gap Report",
            description="Display the gap detection report from the last analysis run.",
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        return types.Property(
            inputs,
            view=types.View(label="Show Gap Report"),
        )

    def execute(self, ctx):
        dataset = ctx.dataset
        gap_report = dataset.info.get("gap_report", None)

        if gap_report is None:
            return {
                "report": "**No gap report found.** Run 'Analyze Coverage' first.",
                "coverage_score": 0.0,
                "n_sparse": 0,
                "n_gaps": 0,
            }

        score = gap_report.get("coverage_score", 0.0)
        sparse = gap_report.get("sparse_clusters", [])
        gaps = gap_report.get("category_gaps", [])

        # Build markdown report
        lines = []
        lines.append(f"## Coverage Score: {score:.2f}")
        lines.append("")

        # Sparse clusters
        if sparse:
            lines.append(f"### Sparse Clusters ({len(sparse)})")
            for sc in sparse:
                label = sc["label"]
                if len(label) > 60:
                    label = label[:60] + "..."
                lines.append(
                    f"- **Cluster {sc['cluster_id']}** "
                    f"({sc['count']} samples): {label}"
                )
        else:
            lines.append("### Sparse Clusters: None")
        lines.append("")

        # Category gaps
        if gaps:
            lines.append(f"### Category Gaps ({len(gaps)})")
            for cg in gaps:
                closest = cg["closest_cluster"]
                if len(closest) > 50:
                    closest = closest[:50] + "..."
                lines.append(
                    f"- **{cg['category']}** "
                    f"(similarity: {cg['similarity']:.2f}, "
                    f"nearest: {closest})"
                )
        else:
            lines.append("### Category Gaps: None")
        lines.append("")

        # Cluster summary from dataset samples
        cluster_summary = {}
        for sample in dataset:
            try:
                cid = sample["cluster_id"]
                label = sample["cluster_label"]
            except (KeyError, AttributeError):
                continue
            if cid is not None:
                if cid not in cluster_summary:
                    cluster_summary[cid] = {
                        "label": label or f"Cluster {cid}",
                        "count": 0,
                    }
                cluster_summary[cid]["count"] += 1

        if cluster_summary:
            lines.append(f"### Cluster Summary ({len(cluster_summary)} clusters)")
            for cid in sorted(cluster_summary.keys()):
                info = cluster_summary[cid]
                label = info["label"]
                if len(label) > 60:
                    label = label[:60] + "..."
                lines.append(
                    f"- **Cluster {cid}** ({info['count']} samples): {label}"
                )

        report_md = "\n".join(lines)

        return {
            "report": report_md,
            "coverage_score": score,
            "n_sparse": len(sparse),
            "n_gaps": len(gaps),
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        outputs.str(
            "report",
            label="Gap Report",
            view=types.MarkdownView(),
        )
        outputs.float("coverage_score", label="Coverage Score")
        outputs.int("n_sparse", label="Sparse Clusters")
        outputs.int("n_gaps", label="Category Gaps")
        return types.Property(
            outputs,
            view=types.View(label="Gap Detection Report"),
        )


# ============================================================
# Plugin registration
# ============================================================

def register(p):
    p.register(AnalyzeCoverage)
    p.register(ShowGapReport)
