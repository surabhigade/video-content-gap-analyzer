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
from scipy.spatial import ConvexHull, Voronoi
from scipy.spatial.qhull import QhullError
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
from .gap_report import (
    HISTORY_INFO_KEY,
    build_coverage_history_entry,
    build_tiered_report,
    compute_coverage_diff,
    render_tiered_report_md,
)
from .report_export import export_coverage_report

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

# Pegasus descriptions — sample up to this many reps per cluster in a
# diverse mix: the 2 closest to the centroid (prototypical), 1 from the
# cluster boundary (atypical for the cluster), and 1 random sample.
PEGASUS_REPS_NEAREST = 2
PEGASUS_REPS_BOUNDARY = 1
PEGASUS_REPS_RANDOM = 1
PEGASUS_REPS_TOTAL = (
    PEGASUS_REPS_NEAREST + PEGASUS_REPS_BOUNDARY + PEGASUS_REPS_RANDOM
)

# Seed the per-cluster random picker so descriptions are reproducible
# across re-runs on the same dataset.
PEGASUS_RANDOM_SEED = 42

INDEX_NAME_PREFIX = "gap-analyzer"

# Structured prompt — the two labeled sections drive two FiftyOne fields:
# ``cluster_label`` holds the aggregated COMMON descriptions, and
# ``cluster_diversity`` holds the aggregated VARIATION notes. Keeping the
# labels explicit lets us parse even when Pegasus runs the two sections
# together on one line.
PEGASUS_PROMPT = (
    "Describe this video in two labeled sections:\n"
    "COMMON: a single concise sentence about the main activity, setting, "
    "and key objects — the kind of detail that would likely hold for other "
    "videos of the same topic.\n"
    "VARIATION: a brief phrase (a few words) naming any distinctive element "
    "that could vary within a group of similar videos — unusual angle, "
    "specific subject detail, lighting, or timing. If nothing stands out, "
    "say 'typical example'."
)

POLL_INTERVAL = 10.0
RATE_LIMIT_WAIT = 30

# Gap detection
SPARSE_THRESHOLD = 3
GAP_SIMILARITY_THRESHOLD = 0.30
ISOLATION_STD_FACTOR = 1.5

# Voronoi coverage — cells smaller than `median_area * VORONOI_CELL_THRESHOLD_FACTOR`
# are treated as "well-covered" (dense region). Coverage is the fraction of the
# convex hull's area these small cells occupy. 1.5 × median keeps the threshold
# self-calibrating: dataset density varies, but the median does too.
VORONOI_CELL_THRESHOLD_FACTOR = 1.5

# Gap-priority scoring — three weights that blend into a [0, 100] score
# attached to every detected gap so the report can sort by importance.
#   distance:  how far (cosine) the gap sits from the nearest cluster
#   expected:  hard yes/no boost for user-listed categories
#   isolation: the nearest cluster's own distance to its nearest peer —
#              gaps in sparse regions are harder to fill and rank higher
# Weights are linearly normalized; the defaults sum to 1.0. Edit these
# constants (or pass ``weights`` to score_gap_priority directly) to rebalance
# what the report treats as most important.
GAP_PRIORITY_WEIGHT_DISTANCE = 0.5
GAP_PRIORITY_WEIGHT_EXPECTED = 0.3
GAP_PRIORITY_WEIGHT_ISOLATION = 0.2

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
    """Pick a diverse set of representatives per (non-noise) cluster.

    For each cluster we select, in order:
      - ``PEGASUS_REPS_NEAREST`` samples closest to the centroid (prototypes)
      - ``PEGASUS_REPS_BOUNDARY`` sample(s) at the cluster's far edge
        (highest centroid distance — atypical-but-still-belongs)
      - ``PEGASUS_REPS_RANDOM`` sample(s) drawn randomly from the rest
        (captures variation the first three might miss)

    Duplicates are skipped — a small cluster just gets however many unique
    samples it can offer. Random draws are seeded per-cluster so the pick
    is reproducible across re-runs on the same dataset.
    """
    import random as _random

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
        n = len(sorted_samples)

        reps = []
        chosen_ids = set()

        def _add(sample) -> None:
            if sample.id not in chosen_ids:
                reps.append(sample)
                chosen_ids.add(sample.id)

        # 2 (or fewer) nearest-centroid samples — the prototype pair
        for i in range(min(PEGASUS_REPS_NEAREST, n)):
            _add(sorted_samples[i][1])

        # 1 boundary sample — highest centroid distance within the cluster
        for i in range(PEGASUS_REPS_BOUNDARY):
            if n - 1 - i < 0:
                break
            _add(sorted_samples[-(i + 1)][1])

        # 1 random sample from anything not yet chosen (deterministic seed)
        remaining = [s for _, s in sorted_samples if s.id not in chosen_ids]
        if remaining and PEGASUS_REPS_RANDOM > 0:
            rng = _random.Random(PEGASUS_RANDOM_SEED + int(cid))
            for _ in range(min(PEGASUS_REPS_RANDOM, len(remaining))):
                pick = rng.choice(remaining)
                _add(pick)
                remaining.remove(pick)

        representatives[cid] = reps

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


def parse_pegasus_response(text: str) -> tuple:
    """Split a Pegasus COMMON/VARIATION labeled response into two strings.

    Tolerates:
      - either label missing
      - labels in any case, with or without markdown bolding
      - multi-line content (lines following a label belong to that label
        until the next label shows up)
      - entirely unlabeled text (falls back to ``(text, "")``)
    """
    if not text:
        return "", ""

    text = text.strip()
    # Strip common markdown affordances Pegasus sometimes emits
    cleaned = text.replace("**", "").replace("__", "")

    buf = {"common": [], "variation": []}
    current = None
    for line in cleaned.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        upper = stripped.upper()
        if upper.startswith("COMMON:"):
            current = "common"
            rest = stripped[len("COMMON:"):].strip()
            if rest:
                buf[current].append(rest)
        elif upper.startswith("VARIATION:"):
            current = "variation"
            rest = stripped[len("VARIATION:"):].strip()
            if rest:
                buf[current].append(rest)
        elif current is not None:
            buf[current].append(stripped)

    common = " ".join(buf["common"]).strip()
    variation = " ".join(buf["variation"]).strip()

    # Fallback: the model ignored labels — treat the whole reply as common.
    if not common and not variation:
        common = cleaned.strip()

    return common, variation


def generate_description(
    client: TwelveLabs, filepath: str, use_indexing: bool, index_id: Optional[str]
) -> tuple:
    """Generate a Pegasus description for a video.

    Returns ``(common, variation, use_indexing_updated)`` where ``common``
    and ``variation`` are the two parsed sections of the Pegasus reply
    (either may be empty). On failure both strings are empty.

    Handles rate limits and approach fallback (A -> B).
    """
    asset = upload_asset(client, filepath)
    if asset is None:
        return "", "", use_indexing

    for attempt in range(2):
        try:
            if use_indexing:
                text = index_and_analyze(client, index_id, asset.id, PEGASUS_PROMPT)
            else:
                text = analyze_via_asset(client, asset.id, PEGASUS_PROMPT)

            if text:
                common, variation = parse_pegasus_response(text)
                return common, variation, use_indexing
            return "", "", use_indexing

        except Exception as e:
            error_str = str(e).lower()

            # Rate limit: wait and retry once
            if "429" in str(e) or "rate" in error_str or "too many" in error_str:
                if attempt == 0:
                    logger.info("Rate limited, waiting %ds...", RATE_LIMIT_WAIT)
                    time.sleep(RATE_LIMIT_WAIT)
                    continue
                return "", "", use_indexing

            # Non-rate-limit error on Approach A: switch to B
            if not use_indexing:
                logger.info("Direct analysis failed, switching to index-based approach")
                return "", "", True

            # Approach B failure
            logger.warning("Description generation failed: %s", e)
            return "", "", use_indexing

    return "", "", use_indexing


def generate_cluster_labels(
    client: TwelveLabs, dataset: fo.Dataset, use_pegasus: bool, ctx
) -> dict:
    """Generate per-cluster descriptions via Pegasus and write two sample fields.

    For each (non-noise) cluster we sample up to ``PEGASUS_REPS_TOTAL`` diverse
    representatives (see ``find_cluster_representatives``), send each one to
    Pegasus with a labeled prompt, and aggregate the results into:

    * ``cluster_label`` — joined COMMON descriptions (what the reps share)
    * ``cluster_diversity`` — joined VARIATION notes (how they differ)

    When ``use_pegasus`` is False, both fields fall back to ``"Cluster N"``
    and an empty string so the downstream UI still has something to show.

    Returns ``{cluster_id: label_str}`` for call-site compatibility; the
    diversity mapping is written directly onto the samples.
    """
    representatives = find_cluster_representatives(dataset)
    cluster_labels = {}
    cluster_diversity = {}
    n_clusters = len(representatives)

    if not use_pegasus:
        # Fast path: generic labels, no diversity info.
        for cid in sorted(representatives.keys()):
            cluster_labels[cid] = f"Cluster {cid}"
            cluster_diversity[cid] = ""
    else:
        use_indexing = False
        index_id = None

        for idx, cid in enumerate(sorted(representatives.keys())):
            ctx.set_progress(
                progress=0.50 + 0.25 * (idx / max(n_clusters, 1)),
                label=f"Generating description for cluster {idx + 1}/{n_clusters}...",
            )

            samples = representatives[cid]
            commons = []
            variations = []

            if use_indexing and index_id is None:
                index_id = create_pegasus_index(client)

            for sample in samples:
                common, variation, use_indexing_new = generate_description(
                    client, sample.filepath, use_indexing, index_id
                )

                # Approach-A → Approach-B switch: if the direct call failed
                # with a non-rate-limit error, retry this sample via indexing.
                if use_indexing_new and not use_indexing:
                    use_indexing = True
                    if index_id is None:
                        index_id = create_pegasus_index(client)
                    common, variation, _ = generate_description(
                        client, sample.filepath, use_indexing, index_id
                    )

                if common:
                    commons.append(common)
                if variation:
                    variations.append(variation)

            # Label = the common descriptions joined (first two get the
            # most weight — they come from the two nearest-centroid reps).
            if commons:
                cluster_labels[cid] = "; ".join(commons[:PEGASUS_REPS_NEAREST]) or commons[0]
            else:
                cluster_labels[cid] = f"Cluster {cid}"

            # Diversity note = the variation phrases from every rep that
            # produced one, deduplicated while preserving order.
            seen = set()
            deduped = []
            for v in variations:
                key = v.lower()
                if key not in seen:
                    seen.add(key)
                    deduped.append(v)
            cluster_diversity[cid] = "; ".join(deduped)

    # Apply both fields to every sample. HDBSCAN noise (cid=-1) never got
    # descriptions above, so tag with a clear marker and an empty diversity.
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
            sample["cluster_diversity"] = ""
            sample.save()
        elif cid in cluster_labels:
            sample["cluster_label"] = cluster_labels[cid]
            sample["cluster_diversity"] = cluster_diversity.get(cid, "")
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


# ------------------------------------------------------------
# Voronoi-based coverage helpers
# ------------------------------------------------------------

def _polygon_area(verts) -> float:
    """Shoelace area of a polygon; orientation-agnostic via abs()."""
    v = np.asarray(verts, dtype=float)
    if v.ndim != 2 or v.shape[0] < 3:
        return 0.0
    x = v[:, 0]
    y = v[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _ensure_ccw(verts) -> np.ndarray:
    """Return the polygon with counter-clockwise winding."""
    v = np.asarray(verts, dtype=float)
    signed = (v[:, 0] * np.roll(v[:, 1], -1) - v[:, 1] * np.roll(v[:, 0], -1)).sum() / 2.0
    return v if signed > 0 else v[::-1]


def _is_left_of_edge(p, a, b) -> bool:
    """Left-of-edge test for a directed CCW edge a->b."""
    return ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])) >= 0.0


def _line_segment_intersect(p1, p2, a, b):
    """Intersect segment p1->p2 with the infinite line through a->b."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = a
    x4, y4 = b
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return p1  # ~parallel — degenerate; caller rarely hits this path
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    return (x1 - t * (x1 - x2), y1 - t * (y1 - y2))


def _clip_polygon_convex(subject, clip_ccw) -> list:
    """Sutherland-Hodgman: clip ``subject`` polygon against convex ``clip_ccw``."""
    output = [tuple(p) for p in subject]
    n = len(clip_ccw)
    for i in range(n):
        if not output:
            break
        a = tuple(clip_ccw[i])
        b = tuple(clip_ccw[(i + 1) % n])
        input_list = output
        output = []
        m = len(input_list)
        for j in range(m):
            p = input_list[j]
            q = input_list[(j + 1) % m]
            p_in = _is_left_of_edge(p, a, b)
            q_in = _is_left_of_edge(q, a, b)
            if p_in:
                output.append(p)
                if not q_in:
                    output.append(_line_segment_intersect(p, q, a, b))
            elif q_in:
                output.append(_line_segment_intersect(p, q, a, b))
    return output


def compute_umap_coverage(
    umap_coords: np.ndarray,
    cell_threshold_factor: float = VORONOI_CELL_THRESHOLD_FACTOR,
) -> float:
    """Voronoi-based density coverage over the convex hull of UMAP points.

    Each point's Voronoi cell (clipped to the hull) is small when the point
    sits in a dense region and large when it's isolated. The coverage score
    is the fraction of the convex hull's area occupied by cells whose area
    is *below* ``median_area × cell_threshold_factor`` — i.e. the
    well-covered, high-density share of the data envelope.

    Returns 0.0 for degenerate inputs (fewer than 4 points, collinear
    points, or any Qhull failure).
    """
    coords = np.asarray(umap_coords, dtype=float)
    n = coords.shape[0]
    if n < 4:
        return 0.0

    try:
        hull = ConvexHull(coords)
    except QhullError as e:
        logger.warning("ConvexHull failed (%s); coverage = 0", e)
        return 0.0

    hull_verts_ccw = _ensure_ccw(coords[hull.vertices])
    hull_area = _polygon_area(hull_verts_ccw)
    if hull_area <= 0.0:
        return 0.0

    # Pad with four far-away corner points so every ORIGINAL point's cell is
    # bounded. Scipy's Voronoi marks boundary-point cells as unbounded (-1
    # vertex index); padding sidesteps that without hand-reconstructing the
    # open edges. The corner points themselves get cells we don't care about.
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    span = max(x_max - x_min, y_max - y_min, 1.0)
    pad = 10.0 * span
    corners = np.array([
        [x_min - pad, y_min - pad],
        [x_min - pad, y_max + pad],
        [x_max + pad, y_min - pad],
        [x_max + pad, y_max + pad],
    ])
    padded = np.vstack([coords, corners])

    try:
        vor = Voronoi(padded)
    except QhullError as e:
        logger.warning("Voronoi failed (%s); coverage = 0", e)
        return 0.0

    cell_areas = []
    for i in range(n):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if not region or -1 in region:
            # Padding should prevent this, but stay safe.
            continue
        cell = vor.vertices[region]
        clipped = _clip_polygon_convex(cell, hull_verts_ccw)
        area = _polygon_area(clipped)
        if area > 0.0:
            cell_areas.append(area)

    if not cell_areas:
        return 0.0

    cell_areas_arr = np.array(cell_areas)
    threshold = float(np.median(cell_areas_arr)) * cell_threshold_factor
    dense_area = float(cell_areas_arr[cell_areas_arr < threshold].sum())

    return float(dense_area / hull_area)


def _clip01(x: float) -> float:
    """Clamp a float into [0.0, 1.0]."""
    return max(0.0, min(1.0, float(x)))


def score_gap_priority(
    centroid_distance: float,
    is_user_expected: bool,
    isolation: float,
    weights: Optional[tuple] = None,
) -> float:
    """Return a 0-100 priority score combining three factors.

    - ``centroid_distance`` (cosine): how far the gap sits from the nearest
      existing cluster. Larger = further from any data we have.
    - ``is_user_expected``: True if this gap corresponds to a category the
      user listed in ``expected_categories`` — a hard yes/no boost for
      user-defined importance.
    - ``isolation`` (cosine): distance between the nearest cluster and its
      own nearest peer cluster. Gaps near an already-sparse region are
      harder to fill, so they should surface higher.

    Weights default to the ``GAP_PRIORITY_WEIGHT_*`` module constants but
    callers can pass an explicit ``(w_distance, w_expected, w_isolation)``
    triple. Weights are linearly normalized, so any proportional triple
    works.
    """
    if weights is None:
        weights = (
            GAP_PRIORITY_WEIGHT_DISTANCE,
            GAP_PRIORITY_WEIGHT_EXPECTED,
            GAP_PRIORITY_WEIGHT_ISOLATION,
        )
    w_d, w_e, w_i = weights
    w_sum = w_d + w_e + w_i
    if w_sum <= 0:
        return 0.0
    raw = (
        w_d * _clip01(centroid_distance)
        + w_e * (1.0 if is_user_expected else 0.0)
        + w_i * _clip01(isolation)
    ) / w_sum
    return round(100.0 * raw, 2)


def compute_cluster_isolation(
    centroids: np.ndarray, unique_ids: np.ndarray
) -> dict:
    """Per-cluster cosine distance to the nearest other cluster centroid.

    Returns ``{cluster_id: float}`` keyed on non-noise cluster IDs. With
    fewer than two clusters there are no "others" to measure against, so
    every entry is 0.0.
    """
    isolation_map = {int(cid): 0.0 for cid in unique_ids}
    n = len(unique_ids)
    if n < 2:
        return isolation_map

    dist_matrix = cosine_distances(centroids, centroids)
    np.fill_diagonal(dist_matrix, np.inf)  # exclude self from the nearest lookup
    for i, cid in enumerate(unique_ids):
        isolation_map[int(cid)] = float(dist_matrix[i].min())
    return isolation_map


def parse_category_hierarchy(text: str) -> list:
    """Parse the hierarchical ``expected_categories`` input string.

    Format: ``"parent1: child1, child2 | parent2: child3, child4"``.
    Pipes separate top-level groups; a colon inside a group splits parent
    from its comma-separated children. A group without a colon is treated
    as a parent whose single leaf *is itself* — which makes the flat legacy
    format (``"a, b, c"``, treated as three groups separated by pipes if
    present, else one parent with children ``a, b, c``) still work.

    Returns a list of ``(parent, [child, ...])`` tuples with at least one
    child per parent. Empty / whitespace-only pieces are dropped.
    """
    if not text or not text.strip():
        return []

    # Split on pipe first; if there are no pipes AND no colons, the legacy
    # flat form ``"a, b, c"`` is a single group of comma-separated leaves.
    # We surface each as its own parent so callers treating every leaf as
    # a top-level category keep working.
    if "|" not in text and ":" not in text:
        leaves = [p.strip() for p in text.split(",") if p.strip()]
        return [(leaf, [leaf]) for leaf in leaves]

    groups = [g.strip() for g in text.split("|") if g.strip()]
    hierarchy = []
    for g in groups:
        if ":" in g:
            parent_str, children_str = g.split(":", 1)
            parent = parent_str.strip()
            children = [c.strip() for c in children_str.split(",") if c.strip()]
        else:
            parent = g.strip()
            children = []
        if not parent:
            continue
        if not children:
            # No explicit children — parent is its own leaf.
            children = [parent]
        hierarchy.append((parent, children))
    return hierarchy


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
    hierarchy: list,
    category_embeddings: dict,
    centroids: np.ndarray,
    unique_ids: np.ndarray,
    cluster_labels_map: dict,
    cluster_umap_centers: dict,
    cluster_isolation_map: dict,
    threshold: float = GAP_SIMILARITY_THRESHOLD,
) -> tuple:
    """Match each leaf category against cluster centroids (not sample points).

    Matching against centroids is the Phase 3.1 design: comparing text against
    per-cluster means is a stabler signal than against individual samples,
    and it makes parent-level aggregation sensible.

    Args:
        hierarchy: [(parent, [child, ...]), ...] from parse_category_hierarchy
        category_embeddings: {leaf_str: np.ndarray(D,)} from embed_categories
        centroids: (K, D) L2-normalized cluster centroids (no -1 / noise)
        unique_ids: (K,) cluster IDs matching ``centroids`` row order
        cluster_labels_map: {cid -> human label}
        cluster_umap_centers: {cid -> (mean_umap_x, mean_umap_y)} — used for
            placing the "Missing" marker near the nearest cluster on the UMAP
        threshold: cosine-similarity floor below which a child is a gap

    Returns (hierarchy_report, flat_gaps).

    hierarchy_report is a list of
        {parent, n_children, n_covered, coverage, best_similarity, children: [...]}
    where each child is
        {category, similarity, closest_cluster, closest_cluster_id,
         is_gap, umap_x, umap_y}.

    flat_gaps is the flat list of missing children (is_gap=True), kept for
    the CoveragePanel's "Missing Categories" scatter trace which expects the
    old shape.
    """
    if len(centroids) == 0 or not category_embeddings:
        return [], []

    # Pre-compute sim(leaf, centroid) for every embedded leaf in one shot.
    leaves = list(category_embeddings.keys())
    leaf_matrix = np.array([category_embeddings[leaf] for leaf in leaves])
    sim_matrix = cosine_similarity(leaf_matrix, centroids)  # (L, K)

    leaf_to_row = {leaf: idx for idx, leaf in enumerate(leaves)}

    def _child_record(leaf: str) -> Optional[dict]:
        row = leaf_to_row.get(leaf)
        if row is None:
            return None
        sims = sim_matrix[row]
        best_col = int(sims.argmax())
        best_cid = int(unique_ids[best_col])
        best_sim = float(sims[best_col])
        umap_xy = cluster_umap_centers.get(best_cid, (0.0, 0.0))
        is_gap = best_sim < threshold
        record = {
            "category": leaf,
            "similarity": round(best_sim, 4),
            "closest_cluster": cluster_labels_map.get(best_cid, f"Cluster {best_cid}"),
            "closest_cluster_id": best_cid,
            "is_gap": is_gap,
            "umap_x": float(umap_xy[0]),
            "umap_y": float(umap_xy[1]),
        }
        # Priority score — only meaningful for entries that actually are gaps.
        # Every leaf here came from the user's expected_categories input, so
        # the "is_user_expected" factor is always True.
        if is_gap:
            centroid_distance = _clip01(1.0 - best_sim)
            isolation = cluster_isolation_map.get(best_cid, 0.0)
            record["priority_score"] = score_gap_priority(
                centroid_distance=centroid_distance,
                is_user_expected=True,
                isolation=isolation,
            )
        return record

    hierarchy_report = []
    flat_gaps = []
    for parent, children in hierarchy:
        child_records = []
        for child in children:
            rec = _child_record(child)
            if rec is None:
                # Embedding failed for this leaf — skip silently; it'll show
                # up as missing from the report rather than crashing the run.
                continue
            child_records.append(rec)

        if not child_records:
            continue

        n_total = len(child_records)
        n_covered = sum(1 for r in child_records if not r["is_gap"])
        best_sim = max(r["similarity"] for r in child_records)

        hierarchy_report.append({
            "parent": parent,
            "n_children": n_total,
            "n_covered": n_covered,
            "coverage": round(n_covered / n_total, 4) if n_total else 0.0,
            "best_similarity": round(best_sim, 4),
            "children": child_records,
        })

        flat_gaps.extend(r for r in child_records if r["is_gap"])

    return hierarchy_report, flat_gaps


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
    """Run full gap detection (structural + hierarchical category-driven).

    ``expected_categories`` here is the parsed hierarchy
    ``[(parent, [child, ...]), ...]`` — parsing happens upstream in the
    operator so the caller can log or surface what it interpreted.

    Returns gap_report dict with both a flat ``category_gaps`` list (for
    Panel compatibility) and a nested ``category_hierarchy`` breakdown.
    """
    ctx.set_progress(progress=0.75, label="Extracting cluster data...")

    samples_list, embeddings, cluster_ids, umap_coords, cluster_labels_map = \
        extract_cluster_data(dataset)

    embeddings_norm = normalize(embeddings, norm="l2")
    centroids, unique_ids = compute_centroids(embeddings_norm, cluster_ids)

    # UMAP mean per cluster — used to place the "Missing Categories" marker
    # on the Panel near the closest cluster's visual center.
    cluster_umap_centers = {}
    for cid in unique_ids:
        mask = cluster_ids == cid
        if mask.any():
            cluster_umap_centers[int(cid)] = (
                float(umap_coords[mask, 0].mean()),
                float(umap_coords[mask, 1].mean()),
            )

    # Per-cluster isolation — distance to the *nearest other* cluster.
    # Feeds the isolation factor of score_gap_priority for both category
    # gaps and sparse clusters.
    cluster_isolation_map = compute_cluster_isolation(centroids, unique_ids)

    # Structural analysis
    ctx.set_progress(progress=0.80, label="Detecting sparse and isolated clusters...")
    sparse_clusters = detect_sparse_clusters(cluster_ids, cluster_labels_map)
    isolated_clusters = detect_isolated_clusters(centroids, unique_ids, cluster_labels_map)
    umap_coverage = compute_umap_coverage(umap_coords)

    # Sparse clusters are "gaps" in the sense that a known region has too
    # few samples. Their centroid-distance factor is 0 (they aren't distant
    # from existing data — they ARE existing data, just thin), so their
    # priority rides on isolation alone. Sparse clusters that also sit far
    # from the rest of the dataset surface to the top.
    for sc in sparse_clusters:
        sc["priority_score"] = score_gap_priority(
            centroid_distance=0.0,
            is_user_expected=False,
            isolation=cluster_isolation_map.get(sc["cluster_id"], 0.0),
        )

    # Category-driven analysis (hierarchical)
    hierarchy_report = []
    flat_gaps = []
    category_coverage = 0.0
    total_leaves = 0
    covered_leaves = 0

    if expected_categories:
        # Flatten the hierarchy into the unique set of leaves we need to embed.
        unique_leaves = list(dict.fromkeys(
            child for _, children in expected_categories for child in children
        ))
        if unique_leaves:
            ctx.set_progress(
                progress=0.85,
                label=f"Embedding {len(unique_leaves)} expected categories...",
            )
            category_embeddings = embed_categories(client, unique_leaves)

            if category_embeddings and len(centroids) > 0:
                ctx.set_progress(progress=0.90, label="Computing category gaps...")
                hierarchy_report, flat_gaps = detect_category_gaps(
                    expected_categories,
                    category_embeddings,
                    centroids,
                    unique_ids,
                    cluster_labels_map,
                    cluster_umap_centers,
                    cluster_isolation_map,
                    threshold=gap_threshold,
                )
                for parent_entry in hierarchy_report:
                    total_leaves += parent_entry["n_children"]
                    covered_leaves += parent_entry["n_covered"]
                if total_leaves:
                    category_coverage = covered_leaves / total_leaves

    # Tag sparse samples
    ctx.set_progress(progress=0.95, label="Tagging sparse cluster samples...")
    sparse_ids = {sc["cluster_id"] for sc in sparse_clusters}
    tag_sparse_samples(dataset, sparse_ids)

    # Compute coverage score
    if total_leaves:
        combined_score = 0.5 * umap_coverage + 0.5 * category_coverage
    else:
        combined_score = umap_coverage

    # Surface highest-priority sparse clusters first so the report highlights
    # the most isolated ones at the top.
    sparse_clusters = sorted(
        sparse_clusters,
        key=lambda sc: sc.get("priority_score", 0.0),
        reverse=True,
    )

    gap_report = {
        "sparse_clusters": sparse_clusters,
        # Flat list — one entry per missing leaf; preserves the Panel's
        # "Missing Categories" trace shape from before 3.1. Sorted by
        # priority_score descending so consumers that only take the top N
        # see the most-important gaps first.
        "category_gaps": sorted(
            (
                {
                    "category": g["category"],
                    "closest_cluster": g["closest_cluster"],
                    "similarity": g["similarity"],
                    "umap_x": g.get("umap_x", 0.0),
                    "umap_y": g.get("umap_y", 0.0),
                    "priority_score": g.get("priority_score", 0.0),
                }
                for g in flat_gaps
            ),
            key=lambda g: g["priority_score"],
            reverse=True,
        ),
        # Hierarchical breakdown: each parent with its coverage fraction and
        # per-child matches. Consumed by ShowGapReport and CoveragePanel.
        "category_hierarchy": hierarchy_report,
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
                "Hierarchical categories to check for coverage. Use a "
                "pipe ('|') between top-level categories and a colon (':') "
                "to list sub-categories under a parent, comma-separated. "
                "Example: 'forklift operations: forward, reverse, loading, "
                "turning | fall hazards: ladder climb, ladder descent, "
                "elevated platform'. Each sub-category is embedded via "
                "Marengo text and matched against cluster centroids, so "
                "you get both parent-level coverage (how much of your "
                "safety taxonomy you cover overall) and child-level gaps "
                "(which specific variant is missing). A flat "
                "comma-separated list still works and is treated as "
                "top-level categories with no children."
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

        # Parse the hierarchical categories string into
        # [(parent, [child, ...]), ...]. Flat legacy inputs degrade to one
        # entry per category with the category itself as its only leaf.
        category_hierarchy = parse_category_hierarchy(expected_categories_str)
        if category_hierarchy:
            n_parents = len(category_hierarchy)
            n_leaves = sum(len(children) for _, children in category_hierarchy)
            logger.info(
                "Expected categories: %d parent(s), %d leaf(-ves) total",
                n_parents, n_leaves,
            )

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
            client, dataset, category_hierarchy, ctx,
            gap_threshold=gap_threshold,
        )

        # Store report
        dataset.info["gap_report"] = gap_report

        # Append a lightweight history entry so subsequent runs can compute
        # a "since last run" diff. Each entry carries timestamp, coverage
        # score, sample + cluster counts, and the list of active gap names —
        # enough for the ShowGapReport / Export operators to tell the user
        # what changed without re-running the whole analysis.
        history = list(dataset.info.get(HISTORY_INFO_KEY, []) or [])
        history.append(build_coverage_history_entry(gap_report, dataset))
        dataset.info[HISTORY_INFO_KEY] = history
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
            # resolve_output will detect this via the "_no_report" marker and
            # render a single Error notice instead of the tiered views.
            return {
                "_no_report": True,
                "report": "**No gap report found.** Run 'Analyze Coverage' first.",
                "coverage_score": 0.0,
                "n_sparse": 0,
                "n_gaps": 0,
            }

        # Compute the "since last run" diff from stored history. The
        # AnalyzeCoverage operator appends a history entry after every run,
        # so the *previous* run is the second-to-last entry. When no prior
        # run exists, diff stays None and the tiered renderer omits the
        # section entirely.
        current_entry = build_coverage_history_entry(gap_report, dataset)
        history = dataset.info.get(HISTORY_INFO_KEY, []) or []
        previous_entry = history[-2] if len(history) >= 2 else None
        diff = compute_coverage_diff(current_entry, previous_entry)

        # Delegate the full three-tier rendering to gap_report so the same
        # data is reusable outside the operator (notebooks, tests).
        tiered = build_tiered_report(gap_report, dataset, diff=diff)
        t1 = tiered["tier1_executive"]
        t2 = tiered["tier2_priority_gaps"]
        t3 = tiered["tier3_full_breakdown"]

        # Priority gap rows, ranked + pre-formatted for TableView display.
        priority_rows = [
            {
                "rank": i + 1,
                "severity": entry["severity"],
                "priority_score": f"{entry['priority_score']:.0f}/100",
                "name": entry["name"],
                "closest_cluster": entry["closest_cluster"],
                "similarity": f"{entry['similarity']:.2f}",
                "collection_recommendation": entry["collection_recommendation"],
            }
            for i, entry in enumerate(t2)
        ]

        # Per-cluster stats table.
        cluster_rows = [
            {
                "cluster_id": c["cluster_id"],
                "size": c["size"],
                "status": (
                    "SPARSE" if c["status"] == "sparse" else "well-covered"
                ),
                "mean_centroid_distance": f"{c['mean_centroid_distance']:.3f}",
                "max_centroid_distance": f"{c['max_centroid_distance']:.3f}",
                "cluster_label": c["cluster_label"] or "",
                "cluster_diversity": c["cluster_diversity"] or "",
            }
            for c in t3["clusters"]
        ]

        # Well-covered categories — flip side of the priority list.
        well_covered_rows = [
            {
                "parent": wc["parent"],
                "category": wc["category"],
                "similarity": f"{wc['similarity']:.2f}",
                "closest_cluster": wc["closest_cluster"],
            }
            for wc in t3["well_covered_categories"]
        ]

        # Pre-format a short, stand-alone diff string for resolve_output so
        # the UI-side code only has to pick a severity, not reformat text.
        diff_label = ""
        diff_description = ""
        if diff is not None:
            delta_sign = "+" if diff["coverage_delta_pct"] >= 0 else ""
            diff_label = (
                f"Since last run ({diff['previous_timestamp']}): "
                f"{diff['coverage_score_prev']*100:.0f}% → "
                f"{diff['coverage_score_curr']*100:.0f}% "
                f"({delta_sign}{diff['coverage_delta_pct']:.1f} pts), "
                f"samples {diff['n_samples_prev']} → {diff['n_samples_curr']}"
            )
            parts = []
            if diff["closed_gaps"]:
                parts.append(f"Closed: {', '.join(diff['closed_gaps'])}")
            if diff["still_open_gaps"]:
                parts.append(f"Still open: {', '.join(diff['still_open_gaps'])}")
            if diff["newly_opened_gaps"]:
                parts.append(f"Newly opened: {', '.join(diff['newly_opened_gaps'])}")
            if not parts:
                parts.append("No gap changes.")
            diff_description = " · ".join(parts)

        return {
            "_no_report": False,
            # Diff ("since last run") — optional, only non-empty when a
            # previous run's history entry is available.
            "has_diff": diff is not None,
            "diff": diff,
            "diff_label": diff_label,
            "diff_description": diff_description,
            "diff_trend": (
                "up" if diff and diff["coverage_delta"] > 0
                else "down" if diff and diff["coverage_delta"] < 0
                else "flat"
            ),
            # Tier 1 fields
            "coverage_pct": t1["coverage_pct"],
            "coverage_quality": t1["coverage_quality"],
            "coverage_label": f"Coverage: {t1['coverage_pct']:.1f}% — {t1['coverage_quality']}",
            "recommendation": t1["recommendation"],
            "total_videos": t1["total_videos"],
            "total_clusters": t1["total_clusters"],
            "n_gaps_total": t1["n_gaps_total"],
            "n_category_gaps": t1["n_category_gaps"],
            "n_sparse_clusters": t1["n_sparse_clusters"],
            # Tier 2 / 3 tables
            "priority_rows": priority_rows,
            "cluster_rows": cluster_rows,
            "well_covered_rows": well_covered_rows,
            "n_outliers": t3["n_outliers"],
            # Full markdown (for programmatic consumers, not rendered in panel)
            "report": render_tiered_report_md(tiered),
            # Legacy keys expected by existing callers
            "coverage_score": t1["coverage_score"],
            "n_sparse": t1["n_sparse_clusters"],
            "n_gaps": t1["n_gaps_total"],
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        result = ctx.results or {}

        # -------- no-report short-circuit --------
        if result.get("_no_report") or not result.get("coverage_quality"):
            outputs.view(
                "no_report",
                types.Error(
                    label="No gap report available",
                    description=(
                        "Run the 'Analyze Coverage' operator on this dataset "
                        "first to generate a report, then re-open 'Show Gap "
                        "Report'."
                    ),
                ),
            )
            return types.Property(
                outputs,
                view=types.View(label="Gap Detection Report"),
            )

        # -------- Since last run (only rendered when history exists) --------
        # Pinned above everything else so the user sees movement before the
        # current-state headline. Severity tracks the coverage trend:
        # Success when up, Warning when down, neutral Notice when flat.
        if result.get("has_diff"):
            trend = result.get("diff_trend", "flat")
            diff_label = result.get("diff_label", "")
            diff_desc = result.get("diff_description", "")
            if trend == "up":
                diff_cls = types.Success
            elif trend == "down":
                diff_cls = types.Warning
            else:
                diff_cls = types.Notice
            outputs.view(
                "since_last_run",
                diff_cls(label=diff_label, description=diff_desc),
            )

        # -------- Tier 1 — Executive Summary --------
        outputs.view(
            "tier1_header",
            types.Header(
                label="Tier 1 — Executive Summary",
                divider=True,
            ),
        )

        quality = result["coverage_quality"]
        label = result.get("coverage_label", "")
        desc = result.get("recommendation", "")
        # Severity-coded notice — surface the coverage band visually so the
        # user reads red / yellow / green before anything else.
        if quality == "Poor":
            notice_cls = types.Error
        elif quality == "Fair":
            notice_cls = types.Warning
        else:  # Good or Excellent
            notice_cls = types.Success
        outputs.view(
            "coverage_notice",
            notice_cls(label=label, description=desc),
        )

        # Quick numeric stats rendered as labelled ints so the panel shows
        # them in a clean vertical stack without markdown formatting quirks.
        outputs.int("total_videos", label="Videos analyzed")
        outputs.int("total_clusters", label="Clusters found")
        outputs.int("n_gaps_total", label="Gaps detected (total)")
        outputs.int("n_category_gaps", label="  ↳ Category gaps")
        outputs.int("n_sparse_clusters", label="  ↳ Sparse clusters")

        # -------- Tier 2 — Priority Gaps --------
        outputs.view(
            "tier2_header",
            types.Header(
                label="Tier 2 — Priority Gaps",
                divider=True,
            ),
        )

        if result.get("priority_rows"):
            priority_table = types.TableView()
            priority_table.add_column("rank", label="#")
            priority_table.add_column("severity", label="Severity")
            priority_table.add_column("priority_score", label="Priority")
            priority_table.add_column("name", label="Gap")
            priority_table.add_column("closest_cluster", label="Closest Cluster")
            priority_table.add_column("similarity", label="Similarity")
            priority_table.add_column(
                "collection_recommendation", label="Recommended action"
            )
            outputs.list(
                "priority_rows",
                types.Object(),
                view=priority_table,
                label="Top gaps ranked by priority (Critical > 70, Moderate 40–70, Low < 40)",
            )
        else:
            outputs.view(
                "no_priority",
                types.Success(
                    label="No gaps detected",
                    description=(
                        "Either no expected categories were provided or the "
                        "dataset already covers every requested category "
                        "within the similarity threshold."
                    ),
                ),
            )

        # -------- Tier 3 — Full Breakdown --------
        outputs.view(
            "tier3_header",
            types.Header(
                label="Tier 3 — Full Breakdown",
                divider=True,
            ),
        )

        if result.get("cluster_rows"):
            cluster_table = types.TableView()
            cluster_table.add_column("cluster_id", label="Cluster")
            cluster_table.add_column("size", label="Size")
            cluster_table.add_column("status", label="Status")
            cluster_table.add_column("mean_centroid_distance", label="Mean dist")
            cluster_table.add_column("max_centroid_distance", label="Max dist")
            cluster_table.add_column("cluster_label", label="Pegasus description")
            cluster_table.add_column("cluster_diversity", label="Diversity note")
            outputs.list(
                "cluster_rows",
                types.Object(),
                view=cluster_table,
                label="Per-cluster statistics",
            )

        if result.get("n_outliers"):
            outputs.view(
                "outliers_notice",
                types.Notice(
                    label=f"{result['n_outliers']} outlier videos (HDBSCAN noise)",
                    description=(
                        "Videos the clustering couldn't assign to any cluster. "
                        "Filter by the 'is_outlier' sample field or "
                        "'sparse_cluster' tag to inspect them."
                    ),
                ),
            )

        if result.get("well_covered_rows"):
            wc_table = types.TableView()
            wc_table.add_column("parent", label="Parent")
            wc_table.add_column("category", label="Category")
            wc_table.add_column("similarity", label="Similarity")
            wc_table.add_column("closest_cluster", label="Nearest cluster")
            outputs.list(
                "well_covered_rows",
                types.Object(),
                view=wc_table,
                label="Well-covered categories (no action needed)",
            )

        return types.Property(
            outputs,
            view=types.View(label="Gap Detection Report"),
        )


# ============================================================
# Operator 3: ExportCoverageReport
# ============================================================

class ExportCoverageReport(foo.Operator):
    """Write a self-contained HTML report to disk.

    The HTML embeds the full three-tier text report, an inline SVG UMAP
    scatter plot built from per-sample umap_x/umap_y fields, and a base64
    data-URI download link for a per-cluster CSV. Recipients without
    FiftyOne installed can open the file in any browser.
    """

    @property
    def config(self):
        return foo.OperatorConfig(
            name="export_coverage_report",
            label="Export Coverage Report",
            description=(
                "Export the gap analysis as a single-file HTML report "
                "(UMAP plot + three-tier report + downloadable CSV). "
                "Run 'Analyze Coverage' first to populate the gap report."
            ),
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.str(
            "output_path",
            label="Output HTML path",
            description=(
                "Absolute or ~-relative path where the self-contained "
                "HTML file will be written. A '.html' suffix is added if "
                "missing. Parent directories are created. Existing files "
                "at the target path are overwritten."
            ),
            default="~/coverage_report.html",
            required=True,
        )
        return types.Property(
            inputs,
            view=types.View(label="Export Coverage Report"),
        )

    def execute(self, ctx):
        dataset = ctx.dataset
        gap_report = dataset.info.get("gap_report", None)
        if gap_report is None:
            return {
                "error": (
                    "No gap report found on the dataset. Run the "
                    "'Analyze Coverage' operator before exporting."
                ),
            }

        raw_path = (ctx.params.get("output_path") or "").strip()
        if not raw_path:
            return {"error": "Output path is required."}

        # Be forgiving on the extension so the user doesn't get surprised.
        if not raw_path.lower().endswith((".html", ".htm")):
            raw_path += ".html"

        try:
            ctx.set_progress(progress=0.1, label="Rendering three-tier report...")
            summary = export_coverage_report(dataset, gap_report, raw_path)
            ctx.set_progress(progress=1.0, label="Export complete")
        except Exception as e:
            logger.exception("export_coverage_report failed")
            return {"error": f"Export failed: {e}"}

        return {
            "output_path": summary["output_path"],
            "file_size_bytes": summary["size_bytes"],
            "n_samples": summary["n_samples"],
            "n_clusters": summary["n_clusters"],
        }

    def resolve_output(self, ctx):
        outputs = types.Object()
        result = ctx.results or {}

        if "error" in result:
            outputs.view(
                "export_error",
                types.Error(
                    label="Export failed",
                    description=result["error"],
                ),
            )
            return types.Property(outputs, view=types.View(label="Export result"))

        path = result.get("output_path", "")
        size = int(result.get("file_size_bytes", 0))
        n_samples = int(result.get("n_samples", 0))
        n_clusters = int(result.get("n_clusters", 0))

        outputs.view(
            "export_success",
            types.Success(
                label="Report exported",
                description=(
                    f"Wrote {size:,} bytes to {path}\n"
                    f"({n_samples} samples plotted, {n_clusters} clusters in CSV). "
                    "Open the file in any browser to view the report; no "
                    "FiftyOne install required for recipients."
                ),
            ),
        )
        outputs.str("output_path", label="Output path")
        outputs.int("file_size_bytes", label="File size (bytes)")
        outputs.int("n_samples", label="Samples plotted")
        outputs.int("n_clusters", label="Clusters in CSV")
        return types.Property(outputs, view=types.View(label="Export result"))


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

        hierarchy = gap_report.get("category_hierarchy", [])
        if hierarchy:
            lines += [
                "", "### Category Coverage (hierarchical)", "",
                "| Parent | Covered | Coverage | Best Sim |",
                "|--------|---------|----------|----------|",
            ]
            for entry in hierarchy:
                pct = entry["coverage"] * 100
                lines.append(
                    f"| {entry['parent'][:40]} "
                    f"| {entry['n_covered']}/{entry['n_children']} "
                    f"| {pct:.0f}% "
                    f"| {entry['best_similarity']:.2f} |"
                )
            # Per-child detail table right below the parent summary
            lines += [
                "", "| Parent | Sub-category | Status | Similarity | Nearest |",
                "|--------|--------------|--------|------------|---------|",
            ]
            for entry in hierarchy:
                for child in entry["children"]:
                    status = "**MISSING**" if child["is_gap"] else "OK"
                    lines.append(
                        f"| {entry['parent'][:25]} "
                        f"| {child['category'][:25]} "
                        f"| {status} "
                        f"| {child['similarity']:.2f} "
                        f"| {child['closest_cluster'][:25]} |"
                    )
        elif category_gaps:
            # Legacy flat fallback
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
    p.register(ExportCoverageReport)
    p.register(CoveragePanel)
