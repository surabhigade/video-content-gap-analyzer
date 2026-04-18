"""
Video Content Gap Analyzer — FiftyOne Plugin

Identifies coverage gaps in video datasets using Twelve Labs Marengo
embeddings and Pegasus descriptions. Two operators:

  - analyze_coverage: Full pipeline (embed -> cluster -> describe -> detect gaps)
  - show_gap_report: Display the gap report from the last analysis run
"""

import os
import time
import asyncio
import logging
import threading
from typing import Optional
from datetime import datetime
from collections import defaultdict

import numpy as np
import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.operators.types as types
from fiftyone.operators.panel import Panel, PanelConfig
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.preprocessing import normalize
import hdbscan
import umap

from twelvelabs import (
    TwelveLabs,
    AsyncTwelveLabs,
    VideoInputRequest,
    MediaSource,
    TextInputRequest,
)
from twelvelabs.types import VideoContext_AssetId
from twelvelabs.indexes import IndexesCreateRequestModelsItem

from .embedding_cache import EmbeddingCache

logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================

# Clustering — HDBSCAN (default) parameters.
# min_cluster_size matches SPARSE_THRESHOLD: any group smaller than this is
# treated as noise by HDBSCAN anyway, so it's also our floor for "is this a
# cluster at all?" Points HDBSCAN can't assign to any cluster are labeled -1
# and surfaced as is_outlier=True on the sample.
HDBSCAN_MIN_CLUSTER_SIZE = 3
HDBSCAN_METRIC = "cosine"
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"

# KMeans fallback — used when the operator input num_clusters > 0. Keeps the
# distance-threshold outlier heuristic the original pipeline relied on.
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
GAP_SIMILARITY_THRESHOLD = 0.30
ISOLATION_STD_FACTOR = 1.5
COVERAGE_GRID_SIZE = 10

# Embedding concurrency — caps simultaneous Twelve Labs embed calls.
# Matches Marengo's documented rate-limit headroom; raise only if you're
# on a higher tier.
EMBED_CONCURRENCY = 5


# ============================================================
# Helper functions — API setup
# ============================================================

def get_twelvelabs_client() -> TwelveLabs:
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

async def _embed_one_async(
    client_async: AsyncTwelveLabs,
    sample: fo.Sample,
    cache: Optional[EmbeddingCache],
    semaphore: asyncio.Semaphore,
) -> str:
    """Embed one sample via async Marengo. Returns a status string.

    Status values: "api" (newly embedded), "cache_hit" (disk cache hit),
    "already_embedded" (field already set), "failed" (API error). The cache
    and in-memory checks happen *outside* the semaphore so cached samples
    don't consume concurrency slots.
    """
    filepath = sample.filepath
    filename = os.path.basename(filepath)

    try:
        if sample["embedding"] is not None:
            return "already_embedded"
    except (KeyError, AttributeError):
        pass

    if cache is not None:
        cached = cache.get(filepath)
        if cached is not None:
            sample["embedding"] = cached.tolist()
            sample.save()
            return "cache_hit"

    async with semaphore:
        try:
            with open(filepath, "rb") as f:
                asset = await client_async.assets.create(method="direct", file=f)

            response = await client_async.embed.v_2.create(
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

            if cache is not None:
                try:
                    cache.put(filepath, embedding)
                except Exception as e:
                    logger.warning(
                        "Failed to cache embedding for %s: %s", filename, e
                    )

            return "api"

        except Exception as e:
            logger.warning("Failed to embed %s: %s", filename, e)
            return "failed"


async def _embed_all_async(dataset: fo.Dataset, ctx) -> tuple:
    """Async orchestration for the embedding stage (0.00-0.25 of the run)."""
    samples = list(dataset)
    total = len(samples)
    semaphore = asyncio.Semaphore(EMBED_CONCURRENCY)

    counts = {"api": 0, "cache_hit": 0, "already_embedded": 0, "failed": 0}
    completed = 0

    api_key = os.environ.get("TWELVELABS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TWELVELABS_API_KEY environment variable is not set."
        )

    async with AsyncTwelveLabs(api_key=api_key) as client_async:
        with EmbeddingCache() as cache:
            tasks = [
                asyncio.create_task(
                    _embed_one_async(client_async, s, cache, semaphore)
                )
                for s in samples
            ]

            # Update progress on each completion so the bar advances even
            # though tasks finish out of order. Using as_completed instead of
            # a loop counter is what makes progress correct under concurrency.
            for coro in asyncio.as_completed(tasks):
                result = await coro
                counts[result] += 1
                completed += 1
                ctx.set_progress(
                    progress=0.25 * (completed / max(total, 1)),
                    label=(
                        f"Embedding {completed}/{total} "
                        f"({counts['api']} new, "
                        f"{counts['cache_hit']} from cache, "
                        f"{counts['already_embedded']} already embedded, "
                        f"{counts['failed']} failed)"
                    ),
                )

    dataset.save()
    return (
        counts["api"],
        counts["failed"],
        counts["cache_hit"] + counts["already_embedded"],
        counts["cache_hit"],
    )


def embed_all_samples(dataset: fo.Dataset, ctx) -> tuple:
    """Synchronous entry point used by AnalyzeCoverage.execute.

    Wraps the async pipeline with ``asyncio.run`` unless we're already inside
    a running event loop (some FiftyOne execution modes), in which case we
    run it in a fresh thread with its own loop to avoid reentrancy errors.

    Returns (api_count, failed_count, skipped_count, cache_hit_count).
    ``skipped_count`` includes both in-memory skips (sample already had the
    embedding field set) and disk-cache hits; ``cache_hit_count`` breaks
    out the latter for logging.
    """
    try:
        asyncio.get_running_loop()
        in_running_loop = True
    except RuntimeError:
        in_running_loop = False

    if not in_running_loop:
        return asyncio.run(_embed_all_async(dataset, ctx))

    result: dict = {}

    def _runner() -> None:
        try:
            result["ok"] = asyncio.run(_embed_all_async(dataset, ctx))
        except BaseException as e:
            result["err"] = e

    t = threading.Thread(target=_runner, name="embed-async-runner")
    t.start()
    t.join()

    if "err" in result:
        raise result["err"]
    return result["ok"]


# ============================================================
# Helper functions — Clustering (from notebook 02)
# ============================================================

def run_clustering(
    dataset: fo.Dataset,
    n_clusters: int = 0,
    outlier_std_factor: float = OUTLIER_STD_FACTOR,
) -> tuple:
    """Cluster the dataset's embeddings with HDBSCAN or KMeans, then UMAP.

    The ``n_clusters`` argument selects the algorithm:
      * ``0`` (default) — HDBSCAN (density-based). Discovers clusters of
        varying sizes, labels low-density points as noise (cluster_id=-1),
        and writes ``cluster_confidence`` from its probabilities_ array.
      * ``> 0`` — KMeans with that exact k. Every sample gets a cluster
        (no noise label). Outliers are flagged by the classic heuristic
        ``distance > mean + outlier_std_factor * std``; confidence is set
        to 1.0 uniformly since KMeans has no native membership score.

    Writes cluster_id, centroid_distance, is_outlier, cluster_confidence,
    umap_x, umap_y to each sample. Returns (n_samples, n_clusters, n_noise).
    """
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

    if n_clusters > 0:
        # -------- KMeans fallback (explicit k from the user) --------
        k = int(n_clusters)
        if k > n_samples:
            logger.warning(
                "num_clusters (%d) > n_samples (%d), clamping", k, n_samples
            )
            k = max(1, n_samples - 1) if n_samples > 1 else 1

        if k <= 1:
            labels = np.zeros(n_samples, dtype=int)
            centroids_norm = normalize(
                embeddings_norm.mean(axis=0, keepdims=True), norm="l2"
            )
        else:
            kmeans = KMeans(
                n_clusters=k,
                random_state=KMEANS_RANDOM_STATE,
                n_init=KMEANS_N_INIT,
            )
            labels = kmeans.fit_predict(embeddings_norm)
            centroids_norm = normalize(kmeans.cluster_centers_, norm="l2")

        distances = np.array([
            cosine_distances(
                embeddings_norm[i:i + 1],
                centroids_norm[labels[i]:labels[i] + 1],
            )[0, 0]
            for i in range(n_samples)
        ])

        std_dist = distances.std()
        if std_dist > 0:
            threshold = distances.mean() + outlier_std_factor * std_dist
            is_outlier = distances > threshold
        else:
            is_outlier = np.zeros(n_samples, dtype=bool)

        # KMeans has no native membership score; 1.0 across the board signals
        # "this algorithm doesn't distinguish confidence" without requiring
        # downstream code to special-case a missing field.
        probabilities = np.ones(n_samples, dtype=float)
    else:
        # -------- HDBSCAN default (density-based) --------
        # HDBSCAN needs at least min_cluster_size samples to find anything;
        # below that the library would raise. Short-circuit with a single
        # trivial cluster and maximal confidence.
        if n_samples < HDBSCAN_MIN_CLUSTER_SIZE:
            labels = np.zeros(n_samples, dtype=int)
            probabilities = np.ones(n_samples, dtype=float)
        else:
            # algorithm="generic" routes through sklearn.pairwise_distances,
            # which is the only path that supports metric="cosine". The default
            # BallTree backend rejects cosine outright.
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
                metric=HDBSCAN_METRIC,
                cluster_selection_method=HDBSCAN_CLUSTER_SELECTION_METHOD,
                algorithm="generic",
            )
            labels = clusterer.fit_predict(embeddings_norm)
            # probabilities_[i] is membership strength in [0, 1] for the
            # assigned cluster; noise points get 0. Low-but-nonzero values
            # mark samples sitting near a cluster boundary — a softer gap
            # signal than -1.
            probabilities = np.asarray(clusterer.probabilities_, dtype=float)

        unique_non_noise = np.sort(np.unique(labels[labels >= 0]))

        # Mean centroid per non-noise cluster, L2-renormalized so cosine
        # distance stays well-defined. Noise points get the distance to the
        # *nearest* centroid — gives the Panel a meaningful "how far from any
        # cluster" size.
        distances = np.zeros(n_samples, dtype=float)
        if len(unique_non_noise) > 0:
            centroid_by_cid = {}
            for cid in unique_non_noise:
                mask = labels == cid
                c = embeddings_norm[mask].mean(axis=0, keepdims=True)
                centroid_by_cid[int(cid)] = normalize(c, norm="l2")[0]

            centroid_matrix = np.stack(list(centroid_by_cid.values()))
            all_dists = cosine_distances(embeddings_norm, centroid_matrix)

            cid_to_col = {cid: i for i, cid in enumerate(centroid_by_cid.keys())}
            for i in range(n_samples):
                if labels[i] >= 0:
                    distances[i] = float(all_dists[i, cid_to_col[int(labels[i])]])
                else:
                    distances[i] = float(all_dists[i].min())

        is_outlier = labels == -1

    # Shared tail: recompute summary counts and run UMAP.
    unique_non_noise = np.sort(np.unique(labels[labels >= 0]))
    n_clusters = int(len(unique_non_noise))
    n_noise = int(np.sum(labels == -1))

    # UMAP 2D reduction. Needs n_neighbors >= 2 and at least 3 samples to
    # form a meaningful manifold; below that we fall back to zero coords so
    # downstream code that reads umap_x/umap_y still gets valid floats.
    if n_samples < 3:
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

    for i, sample in enumerate(samples_list):
        sample["cluster_id"] = int(labels[i])
        sample["centroid_distance"] = float(distances[i])
        sample["is_outlier"] = bool(is_outlier[i])
        sample["cluster_confidence"] = float(probabilities[i])
        sample["umap_x"] = float(coords_2d[i, 0])
        sample["umap_y"] = float(coords_2d[i, 1])
        sample.save()

    dataset.save()
    return n_samples, n_clusters, n_noise


# ============================================================
# Helper functions — Pegasus descriptions (from notebook 03)
# ============================================================

def find_cluster_representatives(dataset: fo.Dataset) -> dict:
    """For each cluster_id, find the REPS_PER_CLUSTER samples closest to centroid."""
    clusters = defaultdict(list)

    for sample in dataset:
        try:
            cid = sample["cluster_id"]
            dist = sample["centroid_distance"]
        except (KeyError, AttributeError):
            continue
        # Skip HDBSCAN noise — describing "the stuff that didn't cluster" is
        # never useful and would waste a Pegasus call on unrelated videos.
        if cid is not None and cid != -1 and dist is not None:
            clusters[cid].append((dist, sample))

    representatives = {}
    for cid in sorted(clusters.keys()):
        sorted_samples = sorted(clusters[cid], key=lambda x: x[0])
        representatives[cid] = [s for _, s in sorted_samples[:REPS_PER_CLUSTER]]

    return representatives


def upload_asset(client: TwelveLabs, filepath: str) -> Optional[object]:
    """Upload a video file as a Twelve Labs asset. Returns asset or None."""
    try:
        with open(filepath, "rb") as f:
            asset = client.assets.create(method="direct", file=f)
        return asset
    except Exception as e:
        logger.warning("Upload failed for %s: %s", os.path.basename(filepath), e)
        return None


def analyze_via_asset(client: TwelveLabs, asset_id: str, prompt: str) -> str:
    """Approach A: analyze directly using asset_id (no indexing)."""
    response = client.analyze(
        video=VideoContext_AssetId(asset_id=asset_id),
        prompt=prompt,
        temperature=0.2,
    )
    return response.data


def create_pegasus_index(client: TwelveLabs) -> str:
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


def index_and_analyze(
    client: TwelveLabs, index_id: str, asset_id: str, prompt: str
) -> str:
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


def generate_description(
    client: TwelveLabs, filepath: str, use_indexing: bool, index_id: Optional[str]
) -> tuple:
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


def generate_cluster_labels(
    client: TwelveLabs, dataset: fo.Dataset, use_pegasus: bool, ctx
) -> dict:
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
                label=f"Generating description for cluster {idx + 1}/{n_clusters}...",
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

    # Apply labels to all samples. HDBSCAN noise points (cid=-1) never get a
    # description pass above, so tag them with a clear marker instead of
    # leaving the field blank.
    NOISE_LABEL = "Outliers (HDBSCAN noise)"
    for sample in dataset:
        try:
            cid = sample["cluster_id"]
        except (KeyError, AttributeError):
            continue
        if cid is None:
            continue
        if cid == -1:
            sample["cluster_label"] = NOISE_LABEL
            sample.save()
        elif cid in cluster_labels:
            sample["cluster_label"] = cluster_labels[cid]
            sample.save()

    dataset.save()
    return cluster_labels


# ============================================================
# Helper functions — Gap detection (from notebook 04)
# ============================================================

def extract_cluster_data(dataset: fo.Dataset) -> tuple:
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


def compute_centroids(
    embeddings_norm: np.ndarray, cluster_ids: np.ndarray
) -> tuple:
    """Recompute L2-normalized cluster centroids from sample embeddings.

    Noise samples (cluster_id = -1) are excluded — there is no meaningful
    centroid for "everything that didn't cluster".
    """
    all_ids = np.unique(cluster_ids)
    unique_ids = np.sort(all_ids[all_ids >= 0])
    centroids = []

    for cid in unique_ids:
        mask = cluster_ids == cid
        centroid = embeddings_norm[mask].mean(axis=0)
        centroids.append(centroid)

    if not centroids:
        return np.zeros((0, embeddings_norm.shape[1])), unique_ids

    centroids = np.array(centroids)
    centroids = normalize(centroids, norm="l2")
    return centroids, unique_ids


def detect_sparse_clusters(
    cluster_ids: np.ndarray, cluster_labels_map: dict, threshold: int = SPARSE_THRESHOLD
) -> list:
    """Flag clusters with fewer than threshold samples.

    HDBSCAN noise (cluster_id = -1) is skipped — noise is already surfaced
    via ``is_outlier`` and the "Outliers" cluster label; calling it "sparse"
    on top of that would double-count.
    """
    unique, counts = np.unique(cluster_ids, return_counts=True)
    sparse = []

    for cid, cnt in zip(unique, counts):
        if cid == -1:
            continue
        if cnt < threshold:
            sparse.append({
                "cluster_id": int(cid),
                "label": cluster_labels_map.get(int(cid), f"Cluster {cid}"),
                "count": int(cnt),
            })

    return sparse


def detect_isolated_clusters(
    centroids: np.ndarray, unique_ids: np.ndarray, cluster_labels_map: dict
) -> list:
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


def compute_umap_coverage(
    umap_coords: np.ndarray, grid_size: int = COVERAGE_GRID_SIZE
) -> float:
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


def embed_categories(client: TwelveLabs, categories: list) -> dict:
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


def detect_category_gaps(
    category_embeddings: dict,
    embeddings_norm: np.ndarray,
    cluster_ids: np.ndarray,
    unique_ids: np.ndarray,
    cluster_labels_map: dict,
    threshold: float = GAP_SIMILARITY_THRESHOLD,
    umap_coords: Optional[np.ndarray] = None,
) -> list:
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

        result = {
            "category": category,
            "closest_cluster": closest_label,
            "closest_cluster_id": closest_cid,
            "similarity": round(max_sim, 4),
            "is_gap": max_sim < threshold,
        }

        if umap_coords is not None:
            result["umap_x"] = float(umap_coords[closest_sample_idx, 0])
            result["umap_y"] = float(umap_coords[closest_sample_idx, 1])

        results.append(result)

    return results


def tag_sparse_samples(dataset: fo.Dataset, sparse_cluster_ids: set) -> int:
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


def detect_gaps(
    client: TwelveLabs,
    dataset: fo.Dataset,
    expected_categories: list,
    ctx,
    gap_threshold: float = GAP_SIMILARITY_THRESHOLD,
) -> dict:
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
                threshold=gap_threshold,
                umap_coords=umap_coords,
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
                "umap_x": cr.get("umap_x", 0.0),
                "umap_y": cr.get("umap_y", 0.0),
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
    """Full pipeline operator: embed videos, cluster, describe, detect gaps."""

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
            label="Number of Clusters (0 = HDBSCAN auto)",
            description=(
                "Leave at 0 (default) to use HDBSCAN: a density-based "
                "algorithm that discovers clusters of varying sizes on its "
                "own and flags low-density videos as outliers. Set to a "
                "positive integer to fall back to KMeans with that exact "
                "number of clusters — useful when you already know your "
                "category count or want a fixed partition."
            ),
            default=0,
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

        inputs.int(
            "max_samples",
            label="Max Samples (0 = all)",
            description=(
                "Maximum number of samples to process. Set to 0 to process all "
                "samples. Useful for large datasets to limit API costs and time."
            ),
            default=0,
        )

        inputs.float(
            "outlier_threshold",
            label="Outlier Threshold (KMeans only, std deviations)",
            description=(
                "Only applied when num_clusters > 0 (KMeans mode). Samples "
                "whose cosine distance to their cluster centroid exceeds "
                "mean + N*std are flagged as outliers. Lower values flag "
                "more outliers. Ignored under HDBSCAN, which labels noise "
                "natively."
            ),
            default=2.0,
        )

        inputs.float(
            "gap_threshold",
            label="Gap Similarity Threshold",
            description=(
                "Categories with max similarity below this value are flagged "
                "as gaps. Higher values flag more gaps."
            ),
            default=0.3,
        )

        inputs.bool(
            "clear_cache",
            label="Clear Embedding Cache (force re-embed)",
            description=(
                "Wipe the persistent embedding cache and every sample's "
                "embedding field before running, so every video gets a fresh "
                "Marengo call. Leave off to reuse cached embeddings."
            ),
            default=False,
        )

        return types.Property(
            inputs,
            view=types.View(label="Video Content Gap Analyzer"),
        )

    def execute(self, ctx):
        dataset = ctx.dataset
        num_clusters = ctx.params.get("num_clusters", 0)
        expected_categories_str = ctx.params.get("expected_categories", "")
        use_pegasus = ctx.params.get("use_pegasus", True)
        max_samples = ctx.params.get("max_samples", 0)
        # outlier_threshold is only consumed by the KMeans branch of
        # run_clustering; HDBSCAN ignores it (noise label handles outliers).
        outlier_threshold = ctx.params.get("outlier_threshold", OUTLIER_STD_FACTOR)
        gap_threshold = ctx.params.get("gap_threshold", 0.3)
        clear_cache = ctx.params.get("clear_cache", False)

        # Parse categories
        expected_categories = [
            c.strip() for c in expected_categories_str.split(",") if c.strip()
        ] if expected_categories_str else []

        # Edge case: empty dataset
        total = len(dataset)
        if total == 0:
            return {"error": "Dataset is empty. Add video samples before analyzing."}

        # Subsample if max_samples is set
        if max_samples > 0 and total > max_samples:
            logger.info(
                "Subsampling %d of %d samples (max_samples=%d)",
                max_samples, total, max_samples,
            )
            dataset = dataset.take(max_samples)
            total = len(dataset)

        # Large dataset warning
        if total > 100:
            logger.warning(
                "Large dataset (%d samples). This may take a while and "
                "incur significant API costs. Consider setting max_samples.",
                total,
            )
            ctx.set_progress(
                progress=0.0,
                label=f"WARNING: {total} samples — this may take a while. "
                "Consider using max_samples to limit processing.",
            )

        # Edge case: very small dataset
        if total < 3:
            logger.warning(
                "Dataset has only %d sample(s); forcing num_clusters=1", total
            )
            num_clusters = 1

        # Validate API key
        try:
            client = get_twelvelabs_client()
        except RuntimeError as e:
            return {"error": str(e)}

        # Optional: wipe both the persistent disk cache and every sample's
        # embedding field so stage 1 hits the API for every video.
        if clear_cache:
            ctx.set_progress(progress=0.0, label="Clearing embedding cache...")
            with EmbeddingCache() as _cache:
                n_cleared = _cache.clear()
            n_fields_wiped = 0
            for _s in dataset:
                try:
                    if _s["embedding"] is not None:
                        _s["embedding"] = None
                        _s.save()
                        n_fields_wiped += 1
                except (KeyError, AttributeError):
                    pass
            logger.info(
                "clear_cache: removed %d disk entries, wiped %d sample embeddings",
                n_cleared, n_fields_wiped,
            )

        # Stage 1: Embeddings (0.00 - 0.25) — runs up to EMBED_CONCURRENCY
        # Twelve Labs calls in parallel via the async SDK; cache lookups
        # short-circuit the API.
        ctx.set_progress(progress=0.0, label="Stage 1/4: Generating video embeddings...")
        success, fail, skip, cache_hits = embed_all_samples(dataset, ctx)
        already_embedded = skip - cache_hits
        logger.info(
            "Embeddings: %d new, %d from disk cache, %d already embedded, %d failed",
            success, cache_hits, already_embedded, fail,
        )

        # Post-stage-1 summary so the cache benefit is visible even when
        # not every sample was cached.
        ctx.set_progress(
            progress=0.25,
            label=(
                f"Embeddings complete: {success} new, {cache_hits} from cache, "
                f"{already_embedded} already embedded"
                + (f", {fail} failed" if fail else "")
            ),
        )

        # Stage 2: Clustering (0.25 - 0.50). num_clusters == 0 (default) uses
        # HDBSCAN and its native noise detection; a positive value falls back
        # to KMeans with that exact k and the classic distance-based outlier
        # heuristic parameterized by ``outlier_threshold``.
        algorithm_label = "HDBSCAN" if num_clusters == 0 else f"KMeans (k={num_clusters})"
        ctx.set_progress(
            progress=0.25,
            label=f"Stage 2/4: Clustering embeddings with {algorithm_label}...",
        )
        n_samples, num_clusters, n_noise = run_clustering(
            dataset,
            n_clusters=num_clusters,
            outlier_std_factor=outlier_threshold,
        )
        ctx.set_progress(
            progress=0.50,
            label=(
                f"Clustered {n_samples - n_noise} samples into {num_clusters} groups"
                + (f" ({n_noise} outliers)" if n_noise else "")
            ),
        )

        # Stage 3: Cluster descriptions (0.50 - 0.75)
        ctx.set_progress(progress=0.50, label="Stage 3/4: Generating cluster descriptions...")
        cluster_labels = generate_cluster_labels(client, dataset, use_pegasus, ctx)

        # Stage 4: Gap detection (0.75 - 1.00)
        ctx.set_progress(progress=0.75, label="Stage 4/4: Detecting coverage gaps...")
        gap_report = detect_gaps(
            client, dataset, expected_categories, ctx,
            gap_threshold=gap_threshold,
        )

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
            "n_fresh_embeddings": success,
            "n_from_cache": cache_hits,
            "n_already_embedded": already_embedded,
            "n_failed_embeddings": fail,
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        result = ctx.results or {}
        if "error" in result:
            outputs.str("error", label="Error")
        outputs.float("coverage_score", label="Coverage Score (0-1)")
        outputs.int("n_sparse_clusters", label="Sparse Clusters Found")
        outputs.int("n_category_gaps", label="Category Gaps Found")
        outputs.int("n_samples", label="Samples Analyzed")
        outputs.int("n_clusters", label="Clusters Created")
        outputs.int("n_fresh_embeddings", label="Embeddings (fresh API calls)")
        outputs.int("n_from_cache", label="Embeddings (loaded from cache)")
        outputs.int("n_already_embedded", label="Embeddings (already on sample)")
        outputs.int("n_failed_embeddings", label="Embedding Failures")
        return types.Property(
            outputs,
            view=types.View(label="Coverage Analysis Complete"),
        )


# ============================================================
# Operator 2: ShowGapReport
# ============================================================

class ShowGapReport(foo.Operator):
    """Display the gap detection report from the last analysis run."""

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
# Panel: CoveragePanel
# ============================================================

class CoveragePanel(Panel):
    """Interactive visualization of video content gap analysis results."""

    @property
    def config(self):
        return PanelConfig(
            name="coverage_panel",
            label="Coverage Map",
            surfaces="grid",
            allow_multiple=False,
        )

    def on_load(self, ctx):
        self._build_panel_data(ctx)

    def on_change_dataset(self, ctx):
        self._build_panel_data(ctx)

    def _build_panel_data(self, ctx):
        """Extract sample data and build plot traces + summary markdown."""
        dataset = ctx.dataset
        if dataset is None:
            ctx.panel.state.has_data = False
            return

        gap_report = dataset.info.get("gap_report")
        if not gap_report:
            ctx.panel.state.has_data = False
            return

        ctx.panel.state.has_data = True
        score = gap_report.get("coverage_score", 0.0)

        # Collect per-sample data. HDBSCAN noise (cluster_id = -1) is split
        # into a dedicated bucket so we can render it as visually distinct
        # gray/hollow markers instead of another color in the cluster rotation.
        clusters = {}
        noise_bucket = {"x": [], "y": [], "ids": [], "text": []}
        max_dist = 0.001

        for sample in dataset:
            try:
                ux = sample["umap_x"]
                uy = sample["umap_y"]
                cid = sample["cluster_id"]
                clabel = sample.get_field("cluster_label") or f"Cluster {cid}"
                cdist = sample["centroid_distance"]
            except (KeyError, AttributeError):
                continue
            if ux is None or uy is None or cid is None:
                continue

            fname = os.path.basename(sample.filepath)

            if cid == -1:
                noise_bucket["x"].append(ux)
                noise_bucket["y"].append(uy)
                noise_bucket["ids"].append(sample.id)
                noise_bucket["text"].append(
                    f"{fname}<br>Outlier (HDBSCAN noise)<br>"
                    f"Nearest-centroid distance: {cdist:.3f}"
                )
                continue

            if cid not in clusters:
                clusters[cid] = {
                    "label": clabel, "x": [], "y": [],
                    "ids": [], "text": [], "dists": [],
                }
            clusters[cid]["x"].append(ux)
            clusters[cid]["y"].append(uy)
            clusters[cid]["ids"].append(sample.id)
            clusters[cid]["text"].append(
                f"{fname}<br>Cluster: {clabel[:40]}<br>Distance: {cdist:.3f}"
            )
            clusters[cid]["dists"].append(cdist)
            if cdist > max_dist:
                max_dist = cdist

        if not clusters and not noise_bucket["x"]:
            ctx.panel.state.has_data = False
            return

        # Build one Plotly trace per cluster
        traces = []
        for cid in sorted(clusters.keys()):
            c = clusters[cid]
            sizes = [6 + 8 * (d / max_dist) for d in c["dists"]]
            label_short = (
                c["label"][:30] + "..." if len(c["label"]) > 30 else c["label"]
            )
            traces.append({
                "type": "scatter",
                "mode": "markers",
                "name": f"C{cid}: {label_short}",
                "x": c["x"],
                "y": c["y"],
                "ids": c["ids"],
                "marker": {"size": sizes, "opacity": 0.75},
                "text": c["text"],
                "hovertemplate": "%{text}<extra></extra>",
            })

        # Noise / outliers — gray hollow circles so they sit behind clusters
        # visually but stay clickable and findable in the legend.
        if noise_bucket["x"]:
            traces.append({
                "type": "scatter",
                "mode": "markers",
                "name": f"Outliers ({len(noise_bucket['x'])})",
                "x": noise_bucket["x"],
                "y": noise_bucket["y"],
                "ids": noise_bucket["ids"],
                "marker": {
                    "size": 9,
                    "color": "rgba(0,0,0,0)",
                    "line": {"color": "rgba(140,140,140,0.9)", "width": 1.5},
                    "symbol": "circle-open",
                },
                "text": noise_bucket["text"],
                "hovertemplate": "%{text}<extra></extra>",
            })

        # Category gap markers (hollow red diamonds)
        category_gaps = gap_report.get("category_gaps", [])
        gaps_with_coords = [
            g for g in category_gaps
            if g.get("umap_x") is not None and g.get("umap_y") is not None
        ]
        if gaps_with_coords:
            traces.append({
                "type": "scatter",
                "mode": "markers",
                "name": "Missing Categories",
                "x": [g["umap_x"] for g in gaps_with_coords],
                "y": [g["umap_y"] for g in gaps_with_coords],
                "marker": {
                    "size": 16,
                    "color": "rgba(0,0,0,0)",
                    "line": {"color": "red", "width": 2.5},
                    "symbol": "diamond-open",
                },
                "text": [
                    f"MISSING: {g['category']}<br>"
                    f"Similarity: {g['similarity']:.2f}<br>"
                    f"Nearest: {g['closest_cluster'][:30]}"
                    for g in gaps_with_coords
                ],
                "hovertemplate": "%{text}<extra></extra>",
            })

        layout = {
            "xaxis": {"title": "UMAP-1", "showgrid": False, "zeroline": False},
            "yaxis": {"title": "UMAP-2", "showgrid": False, "zeroline": False},
            "legend": {"orientation": "h", "y": -0.15},
            "hovermode": "closest",
            "margin": {"l": 40, "r": 20, "t": 10, "b": 60},
            "plot_bgcolor": "rgba(0,0,0,0)",
            "paper_bgcolor": "rgba(0,0,0,0)",
        }

        ctx.panel.state.plot_data = traces
        ctx.panel.state.plot_layout = layout

        # Score markdown
        n_clustered_samples = sum(len(c["x"]) for c in clusters.values())
        n_noise = len(noise_bucket["x"])
        noise_suffix = f" · **{n_noise} outliers**" if n_noise else ""
        ctx.panel.state.score_md = (
            f"# Dataset Coverage: {score:.0%}\n\n"
            f"**{len(clusters)} clusters** across "
            f"**{n_clustered_samples} samples**{noise_suffix}"
        )

        # Summary table
        sparse_ids = {
            s["cluster_id"] for s in gap_report.get("sparse_clusters", [])
        }
        lines = [
            "### Cluster Summary", "",
            "| Cluster | Label | Count | Status |",
            "|---------|-------|-------|--------|",
        ]
        for cid in sorted(clusters.keys()):
            c = clusters[cid]
            count = len(c["x"])
            label = c["label"][:50]
            status = "**Sparse**" if cid in sparse_ids else "OK"
            lines.append(f"| {cid} | {label} | {count} | {status} |")
        if n_noise:
            lines.append(
                f"| — | Outliers (HDBSCAN noise) | {n_noise} | **Outlier** |"
            )

        if category_gaps:
            lines += [
                "", "### Missing Categories", "",
                "| Category | Nearest Cluster | Similarity |",
                "|----------|-----------------|------------|",
            ]
            for g in category_gaps:
                lines.append(
                    f"| {g['category']} "
                    f"| {g['closest_cluster'][:40]} "
                    f"| {g['similarity']:.2f} |"
                )

        ctx.panel.state.summary_md = "\n".join(lines)

    def on_click_scatter(self, ctx):
        """Select the clicked sample in the App grid."""
        sample_id = ctx.params.get("id")
        if sample_id:
            ctx.ops.set_selected_samples([sample_id])

    def render(self, ctx):
        panel = types.Object()
        has_data = ctx.panel.get_state("has_data", False)

        if not has_data:
            panel.md(
                "### No coverage data available\n\n"
                "Run the **Analyze Coverage** operator first to generate "
                "embeddings, clusters, and gap detection results.\n\n"
                "Then reopen this panel to see the visualization.",
                name="no_data_msg",
            )
            return types.Property(panel, view=types.View(label="Coverage Map"))

        # Coverage score header
        panel.md(
            ctx.panel.get_state("score_md", ""),
            name="score_display",
        )

        # Scatter plot
        panel.plot(
            "scatter_plot",
            data=ctx.panel.get_state("plot_data", []),
            layout=ctx.panel.get_state("plot_layout", {}),
            on_click=self.on_click_scatter,
        )

        # Summary table
        panel.md(
            ctx.panel.get_state("summary_md", ""),
            name="summary_table",
        )

        return types.Property(panel, view=types.View(label="Coverage Map"))


# ============================================================
# Plugin registration
# ============================================================

def register(p):
    p.register(AnalyzeCoverage)
    p.register(ShowGapReport)
    p.register(CoveragePanel)
