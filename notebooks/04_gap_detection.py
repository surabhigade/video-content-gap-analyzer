"""
04_gap_detection.py — Detect coverage gaps in the clustered video dataset.

Loads the dataset with embeddings, clusters, and cluster labels from Phases 1-3,
then runs two modes of gap detection:

  Mode A (Structural): Sparse clusters, isolated clusters, UMAP coverage score.
  Mode B (Category-driven): Embed expected categories via Marengo text API,
    compute cosine similarity to cluster centroids, flag missing categories.

Results are stored in dataset.info["gap_report"] and printed to stdout.
"""

import os
import time

import numpy as np
import fiftyone as fo
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from sklearn.preprocessing import normalize
from twelvelabs import TwelveLabs, TextInputRequest

# --- Configuration ---
DATASET_NAME = "Voxel51/Safe_and_Unsafe_Behaviours"
SPARSE_THRESHOLD = 3            # clusters with fewer samples are "sparse"
GAP_SIMILARITY_THRESHOLD = 0.10 # cosine sim below this = gap (calibrated for cross-modal text-video)
ISOLATION_STD_FACTOR = 1.5      # mean + factor*std = isolated
COVERAGE_GRID_SIZE = 10         # NxN grid for UMAP coverage
RATE_LIMIT_WAIT = 30            # seconds to wait on rate limit
EXPECTED_CATEGORIES = [
    "factory floor with industrial machinery",
    "worker in safety helmet operating equipment",
    "person falling",
    "forklift moving",
    "emergency evacuation",
    "fire or smoke in building",
    "machine malfunction",
]


def extract_cluster_data(dataset):
    """Extract embedded + clustered samples from the dataset.

    Returns:
        samples_list: list of FiftyOne Sample objects
        embeddings: np.ndarray (N, 512)
        cluster_ids: np.ndarray (N,) int
        umap_coords: np.ndarray (N, 2)
        cluster_labels_map: dict {cluster_id: label_str}
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

        # Collect cluster labels
        if cid not in cluster_labels_map:
            try:
                label = sample["cluster_label"]
                cluster_labels_map[cid] = label if label else f"Cluster {cid}"
            except (KeyError, AttributeError):
                cluster_labels_map[cid] = f"Cluster {cid}"

    if len(samples_list) == 0:
        raise RuntimeError(
            "No samples have all required fields (embedding, cluster_id, umap_x, umap_y). "
            "Run 01_embeddings.py, 02_clustering.py, and 03_cluster_descriptions.py first."
        )

    embeddings = np.array(embeddings_list)
    cluster_ids = np.array(cluster_ids_list, dtype=int)
    umap_coords = np.array(umap_list)

    return samples_list, embeddings, cluster_ids, umap_coords, cluster_labels_map


def compute_centroids(embeddings_norm, cluster_ids):
    """Recompute L2-normalized cluster centroids from sample embeddings.

    Returns:
        centroids: np.ndarray (K, 512) L2-normalized
        unique_ids: np.ndarray of unique cluster IDs (sorted)
    """
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
    """Flag clusters with fewer than `threshold` samples.

    Returns list of {"cluster_id": int, "label": str, "count": int}.
    """
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
    """Identify clusters whose centroids are far from all others.

    Returns list of {"cluster_id": int, "label": str, "mean_inter_distance": float}.
    Requires >= 3 clusters; returns empty list otherwise.
    """
    n_clusters = len(unique_ids)
    if n_clusters < 3:
        return []

    dist_matrix = cosine_distances(centroids, centroids)
    isolated = []

    mean_dists = []
    for i in range(n_clusters):
        others = [dist_matrix[i, j] for j in range(n_clusters) if j != i]
        mean_dists.append(np.mean(others))

    mean_dists = np.array(mean_dists)
    threshold = mean_dists.mean() + ISOLATION_STD_FACTOR * mean_dists.std()

    for i, cid in enumerate(unique_ids):
        if mean_dists[i] > threshold:
            isolated.append({
                "cluster_id": int(cid),
                "label": cluster_labels_map.get(int(cid), f"Cluster {cid}"),
                "mean_inter_distance": round(float(mean_dists[i]), 4),
            })

    return isolated


def compute_umap_coverage(umap_coords, grid_size=COVERAGE_GRID_SIZE):
    """Compute what fraction of the UMAP bounding box is occupied.

    Divides the bounding box into grid_size x grid_size cells.
    Returns float in [0, 1].
    """
    x_min, x_max = umap_coords[:, 0].min(), umap_coords[:, 0].max()
    y_min, y_max = umap_coords[:, 1].min(), umap_coords[:, 1].max()

    # Add 1% padding to avoid edge issues
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
    Skips failed categories with a warning.
    """
    results = {}

    for i, category in enumerate(categories, start=1):
        print(f"  [{i}/{len(categories)}] Embedding: \"{category}\" ... ", end="", flush=True)
        start = time.time()

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

                elapsed = time.time() - start
                print(f"{len(embedding)}-d ({elapsed:.1f}s)")
                break

            except Exception as e:
                elapsed = time.time() - start
                error_str = str(e).lower()

                if "429" in str(e) or "rate" in error_str or "too many" in error_str:
                    if attempt == 0:
                        print(f"rate limited, waiting {RATE_LIMIT_WAIT}s ... ", end="", flush=True)
                        time.sleep(RATE_LIMIT_WAIT)
                        continue

                print(f"FAILED ({elapsed:.1f}s): {e}")
                break

    return results


def detect_category_gaps(category_embeddings, embeddings_norm, cluster_ids, unique_ids,
                         cluster_labels_map, threshold=GAP_SIMILARITY_THRESHOLD):
    """Compare each category embedding to all individual sample embeddings.

    Uses max similarity to any sample (not just centroids) for the gap decision,
    and reports which cluster the closest sample belongs to.

    Returns list of {"category", "closest_cluster", "closest_cluster_id",
                     "similarity", "is_gap"} for each category.
    """
    categories = list(category_embeddings.keys())
    cat_matrix = np.array([category_embeddings[c] for c in categories])

    # Compare to every individual sample embedding
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
    """Tag samples in sparse clusters with 'sparse_cluster'.

    Returns count of tagged samples.
    """
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


def print_gap_report(sparse_clusters, isolated_clusters, umap_coverage,
                     category_results, combined_score, category_coverage,
                     n_clusters, n_samples):
    """Print a formatted gap detection report to stdout."""
    print()
    print("=" * 60)
    print("GAP DETECTION REPORT")
    print("=" * 60)

    # --- Mode A: Structural ---
    print(f"\nSTRUCTURAL ANALYSIS (Mode A)")
    print(f"  Clusters: {n_clusters} total, {len(sparse_clusters)} sparse, "
          f"{len(isolated_clusters)} isolated")
    print(f"  UMAP coverage: {umap_coverage * 100:.1f}% "
          f"({int(umap_coverage * COVERAGE_GRID_SIZE ** 2)}/{COVERAGE_GRID_SIZE ** 2} "
          f"grid cells occupied)")

    if sparse_clusters:
        print(f"\n  Sparse clusters (< {SPARSE_THRESHOLD} samples):")
        for sc in sparse_clusters:
            label_short = sc["label"][:60] + "..." if len(sc["label"]) > 60 else sc["label"]
            print(f"    Cluster {sc['cluster_id']} ({sc['count']} sample{'s' if sc['count'] != 1 else ''}): "
                  f"\"{label_short}\"")
    else:
        print(f"\n  No sparse clusters (all clusters have >= {SPARSE_THRESHOLD} samples)")

    if isolated_clusters:
        print(f"\n  Isolated clusters:")
        for ic in isolated_clusters:
            print(f"    Cluster {ic['cluster_id']}: mean inter-cluster distance {ic['mean_inter_distance']:.4f}")
    elif n_clusters < 3:
        print(f"\n  Isolation detection: skipped (need >= 3 clusters, have {n_clusters})")
    else:
        print(f"\n  No isolated clusters detected")

    # --- Mode B: Category gaps ---
    if category_results:
        n_gaps = sum(1 for cr in category_results if cr["is_gap"])
        print(f"\nCATEGORY GAP ANALYSIS (Mode B)")
        print(f"  Categories checked: {len(category_results)}")
        print(f"  Gaps found: {n_gaps}")
        print()

        max_cat_len = max(len(cr["category"]) for cr in category_results)
        for cr in category_results:
            status = "GAP" if cr["is_gap"] else "COVERED"
            label_short = cr["closest_cluster"][:50] + "..." if len(cr["closest_cluster"]) > 50 else cr["closest_cluster"]
            print(f"  {cr['category']:<{max_cat_len}} -> {status:>7} "
                  f"(sim: {cr['similarity']:.2f}, nearest: \"{label_short}\")")
    else:
        print(f"\nCATEGORY GAP ANALYSIS (Mode B)")
        print(f"  Skipped (no categories provided or API unavailable)")

    # --- Coverage score ---
    print(f"\nCOVERAGE SCORE: {combined_score:.2f}", end="")
    if category_results:
        print(f"  (structural: {umap_coverage:.2f}, category: {category_coverage:.2f})")
    else:
        print(f"  (structural only)")

    print(f"\nSamples: {n_samples}")
    print(f"Report stored in dataset.info[\"gap_report\"]")
    print("=" * 60)


def main():
    # --- Validate API key (soft — only needed for Mode B) ---
    api_key = os.environ.get("TWELVELABS_API_KEY")

    # --- Load dataset ---
    print(f"Loading dataset: {DATASET_NAME}")
    try:
        dataset = fo.load_dataset(DATASET_NAME)
    except ValueError:
        raise RuntimeError(
            f"Dataset '{DATASET_NAME}' not found in FiftyOne. "
            "Run 01_embeddings.py, 02_clustering.py, and 03_cluster_descriptions.py first."
        )
    print(f"  {len(dataset)} samples\n")

    # --- Validate prerequisite fields ---
    sample = dataset.first()
    for field in ["embedding", "cluster_id", "umap_x", "umap_y"]:
        try:
            val = sample[field]
        except (KeyError, AttributeError):
            raise RuntimeError(
                f"Field '{field}' not found. Run prerequisite scripts first."
            )

    # --- Extract cluster data ---
    print("Extracting cluster data...")
    samples_list, embeddings, cluster_ids, umap_coords, cluster_labels_map = \
        extract_cluster_data(dataset)

    n_samples = len(samples_list)
    n_clusters = len(cluster_labels_map)
    print(f"  {n_samples} samples, {n_clusters} clusters\n")

    # --- L2 normalize and recompute centroids ---
    print("Normalizing embeddings and computing centroids...")
    embeddings_norm = normalize(embeddings, norm="l2")
    centroids, unique_ids = compute_centroids(embeddings_norm, cluster_ids)
    print(f"  Centroid matrix: {centroids.shape}\n")

    # =========================================================
    # MODE A: Structural Gap Analysis
    # =========================================================
    print("MODE A: Structural gap analysis")
    print("-" * 40)

    # Sparse clusters
    sparse_clusters = detect_sparse_clusters(cluster_ids, cluster_labels_map)
    print(f"  Sparse clusters (< {SPARSE_THRESHOLD} samples): {len(sparse_clusters)}")
    for sc in sparse_clusters:
        print(f"    Cluster {sc['cluster_id']}: {sc['count']} samples")

    # Isolated clusters
    isolated_clusters = detect_isolated_clusters(centroids, unique_ids, cluster_labels_map)
    if n_clusters < 3:
        print(f"  Isolated clusters: skipped (need >= 3 clusters, have {n_clusters})")
    else:
        print(f"  Isolated clusters: {len(isolated_clusters)}")
        for ic in isolated_clusters:
            print(f"    Cluster {ic['cluster_id']}: mean distance {ic['mean_inter_distance']:.4f}")

    # UMAP coverage
    umap_coverage = compute_umap_coverage(umap_coords)
    print(f"  UMAP grid coverage: {umap_coverage * 100:.1f}%\n")

    # =========================================================
    # MODE B: Category-Driven Gap Analysis
    # =========================================================
    category_results = []
    category_coverage = 0.0

    if EXPECTED_CATEGORIES:
        if not api_key:
            print("MODE B: Category gap analysis")
            print("-" * 40)
            print("  WARNING: TWELVELABS_API_KEY not set, skipping category analysis\n")
        else:
            print("MODE B: Category gap analysis")
            print("-" * 40)
            client = TwelveLabs(api_key=api_key)

            print("Embedding expected categories via Marengo text API...")
            category_embeddings = embed_categories(client, EXPECTED_CATEGORIES)

            if category_embeddings:
                print(f"\n  {len(category_embeddings)}/{len(EXPECTED_CATEGORIES)} "
                      f"categories embedded successfully")
                category_results = detect_category_gaps(
                    category_embeddings, embeddings_norm, cluster_ids,
                    unique_ids, cluster_labels_map,
                )
                n_covered = sum(1 for cr in category_results if not cr["is_gap"])
                category_coverage = n_covered / len(category_results)
                print(f"  Gaps found: {len(category_results) - n_covered}/{len(category_results)}\n")
            else:
                print("  WARNING: All category embeddings failed, skipping gap analysis\n")

    # =========================================================
    # Tag sparse samples
    # =========================================================
    sparse_ids = {sc["cluster_id"] for sc in sparse_clusters}
    tagged_count = tag_sparse_samples(dataset, sparse_ids)
    if tagged_count > 0:
        print(f"Tagged {tagged_count} samples in sparse clusters with 'sparse_cluster'")

    # =========================================================
    # Build and store gap report
    # =========================================================
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

    dataset.info["gap_report"] = gap_report
    dataset.save()

    # =========================================================
    # Print final report
    # =========================================================
    print_gap_report(
        sparse_clusters, isolated_clusters, umap_coverage,
        category_results, combined_score, category_coverage,
        n_clusters, n_samples,
    )


if __name__ == "__main__":
    main()
