"""
gap_report.py — Tiered report rendering for video coverage analysis.

Consumes the ``gap_report`` dict produced by ``__init__.py::detect_gaps`` and
the FiftyOne dataset that produced it, and emits three complementary views
of the same data, tuned for different audiences:

* **Tier 1 — Executive Summary**: one-line coverage quality + the single most
  urgent action. Designed to read in ten seconds.
* **Tier 2 — Priority Gaps**: ranked table of the top gaps with severity
  labels and concrete collection recommendations.
* **Tier 3 — Full Breakdown**: per-cluster stats + the list of well-covered
  categories that need no action.

Each tier has a pure builder (``build_*``) and is also rolled up by
``build_tiered_report``. ``render_tiered_report_md`` turns the assembled
dict into markdown suitable for a FiftyOne ``MarkdownView``.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Key we stash history under on dataset.info. Exported so callers (the
# analyze operator writes; the show/export operators read) stay in sync.
HISTORY_INFO_KEY = "coverage_history"

# ------------------------------------------------------------
# Quality / severity thresholds
# ------------------------------------------------------------

# Coverage percentage bands — partition 0..100 exclusively on the upper bound.
COVERAGE_POOR_MAX = 30       # < 30        → Poor
COVERAGE_FAIR_MAX = 55       # 30–<55      → Fair
COVERAGE_GOOD_MAX = 80       # 55–<80      → Good
#                            # ≥ 80        → Excellent

# Priority-score severity (0..100). Matches the thresholds called out in the
# phase spec: critical > 70, moderate 40–70, low < 40.
SEVERITY_CRITICAL_MIN = 70
SEVERITY_MODERATE_MIN = 40

# Tier 2 defaults — cap the top list and use severity to tune how much data
# the recommendation suggests collecting.
TIER2_MAX_GAPS = 10
VIDEOS_BY_SEVERITY = {"Critical": 8, "Moderate": 5, "Low": 3}


# ------------------------------------------------------------
# Small pure helpers
# ------------------------------------------------------------

def coverage_quality(score_pct: float) -> str:
    """Bucket a 0..100 coverage percentage into Poor/Fair/Good/Excellent."""
    if score_pct < COVERAGE_POOR_MAX:
        return "Poor"
    if score_pct < COVERAGE_FAIR_MAX:
        return "Fair"
    if score_pct < COVERAGE_GOOD_MAX:
        return "Good"
    return "Excellent"


def severity_label(priority_score: float) -> str:
    """Map a priority_score (0..100) to Critical / Moderate / Low."""
    if priority_score > SEVERITY_CRITICAL_MIN:
        return "Critical"
    if priority_score >= SEVERITY_MODERATE_MIN:
        return "Moderate"
    return "Low"


def _truncate(text: str, max_len: int) -> str:
    if not text:
        return ""
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def build_coverage_history_entry(gap_report: dict, dataset=None) -> dict:
    """Distil a ``gap_report`` + dataset into a small, JSON-friendly record.

    Designed to be append-only on ``dataset.info["coverage_history"]`` so
    each analysis run leaves a trail that subsequent runs can diff against.
    Only the fields needed for the diff summary are kept — heavy artefacts
    (embeddings, hierarchy details) stay in the live ``gap_report``.

    Returns a dict with keys: timestamp, coverage_score, n_samples,
    n_clusters, n_sparse_clusters, gaps (list of {name, priority_score, type}).
    """
    n_samples = 0
    cluster_ids = set()
    if dataset is not None:
        try:
            n_samples = len(dataset)
        except Exception:
            n_samples = 0
        for sample in dataset:
            try:
                cid = sample["cluster_id"]
            except (KeyError, AttributeError):
                continue
            if cid is not None and int(cid) >= 0:
                cluster_ids.add(int(cid))

    category_gaps = gap_report.get("category_gaps", []) or []
    sparse_clusters = gap_report.get("sparse_clusters", []) or []

    gaps = [
        {
            "name": g.get("category", ""),
            "priority_score": float(g.get("priority_score", 0.0)),
            "type": "category",
        }
        for g in category_gaps
    ]
    for s in sparse_clusters:
        cid = s.get("cluster_id")
        # Use the human label when present — cluster IDs are not stable
        # across HDBSCAN reruns but labels derived from representatives
        # give a best-effort handle.
        name = s.get("label") or (f"Sparse cluster {cid}" if cid is not None else "Sparse cluster")
        gaps.append({
            "name": name,
            "priority_score": float(s.get("priority_score", 0.0)),
            "type": "sparse_cluster",
        })

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "coverage_score": float(gap_report.get("coverage_score", 0.0)),
        "n_samples": int(n_samples),
        "n_clusters": len(cluster_ids),
        "n_sparse_clusters": len(sparse_clusters),
        "gaps": gaps,
    }


def compute_coverage_diff(
    current: dict, previous: Optional[dict]
) -> Optional[dict]:
    """Compare two ``coverage_history`` entries and return a diff summary.

    Returns ``None`` when there's no previous entry to compare against.
    The diff keys are:

      previous_timestamp, coverage_delta (0–1 scale) + coverage_delta_pct
      (percentage points), n_samples_prev / n_samples_curr / new_samples,
      closed_gaps / still_open_gaps / newly_opened_gaps (lists of names).

    Gaps are matched by name; this is stable for category gaps (whose names
    come from the user's expected_categories) and best-effort for sparse
    clusters (whose Pegasus-derived labels can drift across runs).
    """
    if previous is None:
        return None

    cur_gap_names = [g.get("name", "") for g in current.get("gaps", []) if g.get("name")]
    prev_gap_names = [g.get("name", "") for g in previous.get("gaps", []) if g.get("name")]
    cur_set = set(cur_gap_names)
    prev_set = set(prev_gap_names)

    coverage_delta = float(current.get("coverage_score", 0.0)) - float(previous.get("coverage_score", 0.0))

    return {
        "previous_timestamp": previous.get("timestamp", ""),
        "coverage_score_prev": float(previous.get("coverage_score", 0.0)),
        "coverage_score_curr": float(current.get("coverage_score", 0.0)),
        "coverage_delta": round(coverage_delta, 4),
        "coverage_delta_pct": round(coverage_delta * 100.0, 1),
        "n_samples_prev": int(previous.get("n_samples", 0)),
        "n_samples_curr": int(current.get("n_samples", 0)),
        "new_samples": int(current.get("n_samples", 0)) - int(previous.get("n_samples", 0)),
        "closed_gaps": sorted(prev_set - cur_set),
        "still_open_gaps": sorted(prev_set & cur_set),
        "newly_opened_gaps": sorted(cur_set - prev_set),
    }


def _top_priority_gap(gap_report: dict) -> Optional[dict]:
    """Single highest-priority gap across both category_gaps and sparse_clusters."""
    category_gaps = gap_report.get("category_gaps", []) or []
    sparse_clusters = gap_report.get("sparse_clusters", []) or []

    candidates = []
    for g in category_gaps:
        candidates.append({"type": "category", **g})
    for s in sparse_clusters:
        candidates.append({"type": "sparse_cluster", **s})

    if not candidates:
        return None
    return max(candidates, key=lambda g: float(g.get("priority_score", 0.0)))


def _format_recommendation(
    top_gap: Optional[dict], coverage_pct: float
) -> str:
    """Build the single actionable sentence surfaced in Tier 1."""
    if top_gap is None:
        if coverage_pct >= COVERAGE_GOOD_MAX:
            return (
                "Coverage is strong — no urgent collection action required; "
                "focus on expanding diversity within existing clusters."
            )
        return (
            "Provide an expected_categories list to the operator to surface "
            "concrete, actionable gap targets."
        )

    score = float(top_gap.get("priority_score", 0.0))
    if top_gap["type"] == "category":
        name = top_gap.get("category", "the highest-priority category")
        closest = top_gap.get("closest_cluster", "")
        closest_snip = f" (nearest existing cluster: '{_truncate(closest, 50)}')" if closest else ""
        return (
            f"Prioritize collecting videos that depict '{name}'"
            f"{closest_snip} — priority {score:.0f}/100, the most urgent gap."
        )

    # sparse_cluster
    label = _truncate(top_gap.get("label", f"cluster {top_gap.get('cluster_id', '?')}"), 50)
    count = int(top_gap.get("count", 0))
    return (
        f"Expand the sparse cluster '{label}' (currently only {count} videos) — "
        f"priority {score:.0f}/100, the most isolated underrepresented region."
    )


# ------------------------------------------------------------
# Tier 1 — Executive Summary
# ------------------------------------------------------------

def build_executive_summary(
    gap_report: dict,
    dataset=None,
    diff: Optional[dict] = None,
) -> dict:
    """Produce the Tier 1 summary dict.

    Reads:
      - gap_report["coverage_score"] (float in [0, 1])
      - gap_report["category_gaps"] and gap_report["sparse_clusters"]
      - dataset for total_videos + total_clusters when provided

    Optional ``diff`` (from ``compute_coverage_diff``) is forwarded on the
    returned summary so downstream renderers can show "since last run"
    deltas before the headline coverage banner.

    Returns a dict with ``coverage_pct``, ``coverage_quality``,
    ``total_videos``, ``total_clusters``, a unified ``n_gaps_total`` count,
    a single-sentence ``recommendation`` pointing at the top-priority gap,
    and ``diff`` (possibly ``None``).
    """
    coverage_score = float(gap_report.get("coverage_score", 0.0))
    coverage_pct = round(coverage_score * 100.0, 1)
    quality = coverage_quality(coverage_pct)

    # Totals from the dataset when available. The FiftyOne Sample iteration
    # is wrapped in try/except because individual samples may be missing
    # cluster_id (e.g. a partial re-run).
    total_videos = 0
    cluster_ids = set()
    if dataset is not None:
        try:
            total_videos = len(dataset)
        except Exception:
            total_videos = 0
        for sample in dataset:
            try:
                cid = sample["cluster_id"]
            except (KeyError, AttributeError):
                continue
            if cid is not None and int(cid) >= 0:
                cluster_ids.add(int(cid))

    total_clusters = len(cluster_ids)
    n_category_gaps = len(gap_report.get("category_gaps", []) or [])
    n_sparse_clusters = len(gap_report.get("sparse_clusters", []) or [])

    top_gap = _top_priority_gap(gap_report)

    return {
        "coverage_score": coverage_score,
        "coverage_pct": coverage_pct,
        "coverage_quality": quality,
        "total_videos": int(total_videos),
        "total_clusters": total_clusters,
        "n_category_gaps": n_category_gaps,
        "n_sparse_clusters": n_sparse_clusters,
        "n_gaps_total": n_category_gaps + n_sparse_clusters,
        "recommendation": _format_recommendation(top_gap, coverage_pct),
        "diff": diff,
    }


# ------------------------------------------------------------
# Tier 2 — Priority Gaps
# ------------------------------------------------------------

def _collection_recommendation(entry: dict) -> str:
    """Concrete 'collect N videos depicting X' phrase for a gap entry."""
    severity = entry.get("severity") or severity_label(entry.get("priority_score", 0.0))
    target = VIDEOS_BY_SEVERITY.get(severity, 5)

    if entry["type"] == "category":
        name = entry.get("name", "this category")
        return (
            f"Collect ~{target} videos depicting '{name}' from varied "
            f"angles, subjects, and settings."
        )

    # sparse_cluster: recommend enough to lift it above the sparse threshold
    current = int(entry.get("count", 0))
    needed = max(target, 5 - current)  # ensure we at least cross the sparse floor
    label = _truncate(entry.get("closest_cluster", ""), 40) or "this cluster"
    return (
        f"Collect ~{needed} more videos similar to '{label}' "
        f"(currently {current}) to densify the cluster."
    )


def build_priority_gaps(
    gap_report: dict, max_gaps: int = TIER2_MAX_GAPS
) -> list:
    """Return the top-N gap entries (category + sparse combined), ranked.

    Each returned entry is a flat dict with:
      type, name, priority_score, severity, closest_cluster, similarity,
      collection_recommendation — plus positional hints (umap_x/umap_y for
      categories, cluster_id/count for sparse clusters).
    """
    category_gaps = gap_report.get("category_gaps", []) or []
    sparse_clusters = gap_report.get("sparse_clusters", []) or []

    entries = []
    for g in category_gaps:
        entries.append({
            "type": "category",
            "name": g.get("category", ""),
            "priority_score": float(g.get("priority_score", 0.0)),
            "closest_cluster": g.get("closest_cluster", ""),
            "similarity": float(g.get("similarity", 0.0)),
            "umap_x": float(g.get("umap_x", 0.0)),
            "umap_y": float(g.get("umap_y", 0.0)),
        })
    for s in sparse_clusters:
        cid = s.get("cluster_id")
        entries.append({
            "type": "sparse_cluster",
            "name": f"Sparse cluster {cid}: {_truncate(s.get('label', ''), 60)}",
            "priority_score": float(s.get("priority_score", 0.0)),
            # For sparse clusters the "closest" IS itself — surface the label
            # so Tier 2 reads consistently with the category-gap rows.
            "closest_cluster": s.get("label", ""),
            "similarity": 1.0,
            "cluster_id": cid,
            "count": int(s.get("count", 0)),
        })

    entries.sort(key=lambda e: e["priority_score"], reverse=True)
    top = entries[:max_gaps]

    for entry in top:
        entry["severity"] = severity_label(entry["priority_score"])
        entry["collection_recommendation"] = _collection_recommendation(entry)

    return top


# ------------------------------------------------------------
# Tier 3 — Full Breakdown
# ------------------------------------------------------------

def build_full_breakdown(gap_report: dict, dataset=None) -> dict:
    """Per-cluster stats + list of well-covered categories.

    For each non-noise cluster we compute size, mean/max centroid distance,
    and whether the plugin flagged it as sparse. Per-sample metadata
    (cluster_label, cluster_diversity) is pulled from any one sample in the
    cluster — it's written identically to all of them.

    Well-covered categories come from ``gap_report["category_hierarchy"]`` —
    every child with ``is_gap == False``.
    """
    clusters_info = []
    n_outliers = 0

    if dataset is not None:
        by_cluster = defaultdict(list)  # {cid: [(sample, dist_or_None)]}
        for sample in dataset:
            try:
                cid = sample["cluster_id"]
                dist = sample["centroid_distance"]
            except (KeyError, AttributeError):
                continue
            if cid is None:
                continue
            by_cluster[int(cid)].append((sample, dist))

        sparse_ids = {
            int(sc["cluster_id"])
            for sc in gap_report.get("sparse_clusters", []) or []
            if sc.get("cluster_id") is not None
        }

        for cid in sorted(by_cluster.keys()):
            rows = by_cluster[cid]
            if cid == -1:
                n_outliers = len(rows)
                continue

            size = len(rows)
            dists = [float(d) for _, d in rows if d is not None]
            mean_d = (sum(dists) / len(dists)) if dists else 0.0
            max_d = max(dists) if dists else 0.0

            label = ""
            diversity = ""
            if rows:
                head = rows[0][0]
                try:
                    label = head.get_field("cluster_label") or ""
                except Exception:
                    pass
                try:
                    diversity = head.get_field("cluster_diversity") or ""
                except Exception:
                    pass

            clusters_info.append({
                "cluster_id": cid,
                "size": size,
                "cluster_label": label,
                "cluster_diversity": diversity,
                "mean_centroid_distance": round(mean_d, 4),
                "max_centroid_distance": round(max_d, 4),
                "status": "sparse" if cid in sparse_ids else "well-covered",
            })

    well_covered = []
    for parent in gap_report.get("category_hierarchy", []) or []:
        for child in parent.get("children", []):
            if not child.get("is_gap", False):
                well_covered.append({
                    "parent": parent.get("parent", ""),
                    "category": child.get("category", ""),
                    "similarity": float(child.get("similarity", 0.0)),
                    "closest_cluster": child.get("closest_cluster", ""),
                })

    return {
        "clusters": clusters_info,
        "n_outliers": n_outliers,
        "well_covered_categories": well_covered,
    }


# ------------------------------------------------------------
# Top-level roll-up + markdown renderer
# ------------------------------------------------------------

def build_tiered_report(
    gap_report: dict,
    dataset=None,
    diff: Optional[dict] = None,
) -> dict:
    """Assemble Tier 1 + Tier 2 + Tier 3 into one dict.

    Pass an optional ``diff`` (usually ``compute_coverage_diff(current, prev)``)
    so the executive summary can display "since last run" deltas at the top.
    """
    return {
        "tier1_executive": build_executive_summary(gap_report, dataset, diff=diff),
        "tier2_priority_gaps": build_priority_gaps(gap_report),
        "tier3_full_breakdown": build_full_breakdown(gap_report, dataset),
    }


def render_tiered_report_md(tiered: dict) -> str:
    """Render the tiered report as a single markdown string."""
    lines: list = []

    t1 = tiered["tier1_executive"]
    t2 = tiered["tier2_priority_gaps"]
    t3 = tiered["tier3_full_breakdown"]

    lines.append("# Gap Analysis Report")
    lines.append("")

    # ------- Since last run (only when we have a diff) -------
    diff = t1.get("diff")
    if diff:
        lines.append(f"## Since last run ({diff.get('previous_timestamp', '')})")
        lines.append("")
        delta_sign = "+" if diff["coverage_delta_pct"] >= 0 else ""
        lines.append(
            f"- **Coverage:** {diff['coverage_score_prev']*100:.1f}% → "
            f"{diff['coverage_score_curr']*100:.1f}% "
            f"({delta_sign}{diff['coverage_delta_pct']:.1f} pts)"
        )
        sign_s = "+" if diff["new_samples"] >= 0 else ""
        lines.append(
            f"- **Samples:** {diff['n_samples_prev']} → "
            f"{diff['n_samples_curr']} ({sign_s}{diff['new_samples']})"
        )
        if diff["closed_gaps"]:
            lines.append(
                f"- ✓ **Closed gaps ({len(diff['closed_gaps'])}):** "
                f"{', '.join(diff['closed_gaps'])}"
            )
        if diff["still_open_gaps"]:
            lines.append(
                f"- ✗ **Still open ({len(diff['still_open_gaps'])}):** "
                f"{', '.join(diff['still_open_gaps'])}"
            )
        if diff["newly_opened_gaps"]:
            lines.append(
                f"- ⚠ **Newly opened ({len(diff['newly_opened_gaps'])}):** "
                f"{', '.join(diff['newly_opened_gaps'])}"
            )
        if not (diff["closed_gaps"] or diff["still_open_gaps"] or diff["newly_opened_gaps"]):
            lines.append("- _No gap changes since the last run._")
        lines.append("")

    # ------- Tier 1 -------
    lines.append("## Tier 1 — Executive Summary")
    lines.append("")
    lines.append(
        f"- **Coverage:** {t1['coverage_pct']:.1f}% "
        f"— **{t1['coverage_quality']}**"
    )
    lines.append(f"- **Videos analyzed:** {t1['total_videos']}")
    lines.append(f"- **Clusters found:** {t1['total_clusters']}")
    lines.append(
        f"- **Gaps detected:** {t1['n_gaps_total']} "
        f"({t1['n_category_gaps']} category, {t1['n_sparse_clusters']} sparse)"
    )
    lines.append(f"- **Recommended next step:** {t1['recommendation']}")
    lines.append("")

    # ------- Tier 2 -------
    lines.append("## Tier 2 — Priority Gaps")
    lines.append("")
    if t2:
        lines.append(
            f"Top {len(t2)} gaps ranked by priority "
            f"(Critical > {SEVERITY_CRITICAL_MIN}, "
            f"Moderate {SEVERITY_MODERATE_MIN}–{SEVERITY_CRITICAL_MIN}, "
            f"Low < {SEVERITY_MODERATE_MIN}):"
        )
        lines.append("")
        lines.append(
            "| # | Severity | Priority | Gap | Closest Cluster | Sim | Recommendation |"
        )
        lines.append(
            "|---|----------|----------|-----|-----------------|-----|----------------|"
        )
        for i, e in enumerate(t2, 1):
            lines.append(
                f"| {i} | **{e['severity']}** | {e['priority_score']:.0f}/100 "
                f"| {_truncate(e['name'], 40)} "
                f"| {_truncate(e['closest_cluster'], 30)} "
                f"| {e['similarity']:.2f} "
                f"| {e['collection_recommendation']} |"
            )
    else:
        lines.append("_No gaps detected._")
    lines.append("")

    # ------- Tier 3 -------
    lines.append("## Tier 3 — Full Breakdown")
    lines.append("")
    if t3["clusters"]:
        lines.append("### Per-cluster statistics")
        lines.append("")
        lines.append(
            "| Cluster | Size | Status | Mean dist | Max dist | Label | Diversity |"
        )
        lines.append(
            "|---------|------|--------|-----------|----------|-------|-----------|"
        )
        for c in t3["clusters"]:
            status = "**SPARSE**" if c["status"] == "sparse" else "OK"
            lines.append(
                f"| {c['cluster_id']} | {c['size']} | {status} "
                f"| {c['mean_centroid_distance']:.3f} "
                f"| {c['max_centroid_distance']:.3f} "
                f"| {_truncate(c['cluster_label'], 40)} "
                f"| {_truncate(c['cluster_diversity'], 40)} |"
            )
        lines.append("")

    if t3["n_outliers"]:
        lines.append(
            f"Outliers (HDBSCAN noise, no cluster assignment): "
            f"**{t3['n_outliers']}** videos."
        )
        lines.append("")

    if t3["well_covered_categories"]:
        lines.append("### Well-covered categories (no action needed)")
        lines.append("")
        lines.append("| Parent | Category | Similarity | Nearest Cluster |")
        lines.append("|--------|----------|------------|-----------------|")
        for wc in t3["well_covered_categories"]:
            lines.append(
                f"| {_truncate(wc['parent'], 25)} | {wc['category']} "
                f"| {wc['similarity']:.2f} "
                f"| {_truncate(wc['closest_cluster'], 30)} |"
            )
    elif not t3["clusters"]:
        lines.append("_No per-cluster data available._")

    return "\n".join(lines)
