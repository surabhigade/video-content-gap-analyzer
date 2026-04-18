#!/usr/bin/env python3
"""
test_pipeline.py — End-to-end integration test for the Video Content Gap Analyzer.

Loads a small subset (12 videos) of the demo dataset, runs the full 4-stage
analysis pipeline twice, and asserts correctness at every checkpoint:

  ✓ All expected sample fields written
  ✓ Coverage score is a float ∈ [0, 1]
  ✓ At least one cluster formed
  ✓ HTML report exported, exists on disk, and is > 1 KB
  ✓ Second run produces a valid historical diff

Requirements:
  - TWELVELABS_API_KEY environment variable (or .env file in project root)
  - Python dependencies from requirements.txt installed
  - Network access (HuggingFace Hub + Twelve Labs API)

Usage:
    export TWELVELABS_API_KEY="your_key"
    python test_pipeline.py
"""

import atexit
import os
import sys
import tempfile
import time
import traceback

# ────────────────────────────────────────────────────────────────────
# 1. Bootstrap — make the FiftyOne plugin importable as "vcga"
# ────────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))

# Load .env so the API key is available even without `export`
_env_path = os.path.join(_DIR, ".env")
if os.path.isfile(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# The project dir uses hyphens (invalid for Python imports) and __init__.py
# uses relative imports. A symlink under a valid name resolves both issues.
_tmp = tempfile.mkdtemp(prefix="vcga_test_")
_link = os.path.join(_tmp, "vcga")
os.symlink(_DIR, _link)
sys.path.insert(0, _tmp)
atexit.register(lambda: (os.unlink(_link), os.rmdir(_tmp)))

import vcga  # noqa: E402  — must come after sys.path manipulation
import numpy as np  # noqa: E402
import fiftyone as fo  # noqa: E402

# ────────────────────────────────────────────────────────────────────
# 2. Configuration
# ────────────────────────────────────────────────────────────────────
DATASET_NAME = "vcga-integration-test"
N_SAMPLES = 12
EXPECTED_CATEGORIES = (
    "safety: hard hat compliance, safety vest "
    "| hazards: fire evacuation, slip and fall"
)
REPORT_PATH = os.path.join(_DIR, "_test_report.html")

# Fields the pipeline must write to every successfully-embedded sample.
REQUIRED_FIELDS = [
    "embedding",
    "cluster_id",
    "cluster_label",
    "cluster_confidence",
    "centroid_distance",
    "is_outlier",
    "umap_x",
    "umap_y",
]


# ────────────────────────────────────────────────────────────────────
# 3. Minimal mock for the FiftyOne operator execution context
# ────────────────────────────────────────────────────────────────────
class Ctx:
    """Stand-in for ``fiftyone.operators.ExecutionContext``."""

    def __init__(self, ds):
        self.dataset = ds

    def set_progress(self, progress=0.0, label=""):
        n = int(progress * 20)
        bar = "█" * n + "░" * (20 - n)
        print(
            f"\r  [{bar}] {progress * 100:5.1f}%  {label[:60]:<60}",
            end="",
            flush=True,
        )
        if progress >= 1.0:
            print()


# ────────────────────────────────────────────────────────────────────
# 4. Test harness — lightweight check() / summary
# ────────────────────────────────────────────────────────────────────
_passed = 0
_failed = 0


def check(description, ok, detail=""):
    """Record a single assertion — prints ✅ or ❌ immediately."""
    global _passed, _failed
    if ok:
        _passed += 1
        print(f"  ✅  {description}")
    else:
        _failed += 1
        suffix = f"  ({detail})" if detail else ""
        print(f"  ❌  {description}{suffix}")


# ────────────────────────────────────────────────────────────────────
# 5. Pipeline runner (mirrors AnalyzeCoverage.execute)
# ────────────────────────────────────────────────────────────────────
def run_pipeline(dataset):
    """Execute stages 1-4 and persist results.

    Returns ``(gap_report_dict, umap_reducer)``.
    """
    ctx = Ctx(dataset)
    client = vcga.get_twelvelabs_client()

    # Stage 1 — Embed
    print("\n  ▸ Stage 1/4  Embedding videos…")
    t = time.time()
    n_new, n_fail, n_skip, n_cache, reasons = vcga.embed_all_samples(dataset, ctx)
    n_reused = n_skip - n_cache
    print(
        f"    {n_new} new · {n_cache} cached · {n_reused} reused · "
        f"{n_fail} failed  ({time.time() - t:.1f}s)"
    )

    # Stage 2 — Cluster
    print("\n  ▸ Stage 2/4  Clustering (HDBSCAN)…")
    t = time.time()
    ns, nc, nn, reducer = vcga.run_clustering(dataset, n_clusters=0)
    print(f"    {nc} clusters · {nn} noise · {ns} total  ({time.time() - t:.1f}s)")

    # Stage 3 — Labels (fast mode — no Pegasus calls)
    print("\n  ▸ Stage 3/4  Labels (fast mode, no Pegasus)…")
    t = time.time()
    vcga.generate_cluster_labels(client, dataset, False, ctx)
    print(f"    done  ({time.time() - t:.1f}s)")

    # Stage 4 — Gap detection
    print("\n  ▸ Stage 4/4  Gap detection…")
    t = time.time()
    hierarchy = vcga.parse_category_hierarchy(EXPECTED_CATEGORIES)
    report = vcga.detect_gaps(
        client, dataset, hierarchy, ctx, umap_reducer=reducer,
    )
    print(
        f"    score={report['coverage_score']:.3f} · "
        f"{len(report['category_gaps'])} category gaps · "
        f"{len(report['sparse_clusters'])} sparse clusters  "
        f"({time.time() - t:.1f}s)"
    )

    # Persist results (same bookkeeping as AnalyzeCoverage.execute)
    dataset.info["gap_report"] = report
    history = list(dataset.info.get(vcga.HISTORY_INFO_KEY, []) or [])
    history.append(vcga.build_coverage_history_entry(report, dataset))
    dataset.info[vcga.HISTORY_INFO_KEY] = history
    dataset.save()

    return report, reducer


# ────────────────────────────────────────────────────────────────────
# 6. Main — load data, run twice, assert everything
# ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 64)
    print("  VIDEO CONTENT GAP ANALYZER — INTEGRATION TEST")
    print("=" * 64)

    # ── Pre-flight ──
    key = os.environ.get("TWELVELABS_API_KEY", "")
    if not key or key == "your_key_here":
        print("\n  ❌  TWELVELABS_API_KEY not set (or still placeholder).")
        print("     Fix with:  export TWELVELABS_API_KEY='tlk_…'")
        sys.exit(1)
    print(f"\n  API key  : …{key[-4:]}")
    print(f"  Dataset  : {DATASET_NAME}  ({N_SAMPLES} samples)")
    print(f"  Categories: {EXPECTED_CATEGORIES}")

    # ── Load / create dataset ──
    if fo.dataset_exists(DATASET_NAME):
        ds = fo.load_dataset(DATASET_NAME)
        print(f"\n  Reusing existing dataset ({len(ds)} samples)")
    else:
        print(f"\n  Downloading {N_SAMPLES} videos from HuggingFace Hub…")
        from fiftyone.utils.huggingface import load_from_hub

        ds = load_from_hub(
            "Voxel51/Safe_and_Unsafe_Behaviours",
            max_samples=N_SAMPLES,
            name=DATASET_NAME,
        )
        print(f"  Downloaded {len(ds)} samples")

    # Reset history so we get a clean two-run diff test
    ds.info.pop(vcga.HISTORY_INFO_KEY, None)
    ds.info.pop("gap_report", None)
    ds.save()

    # ═══════════════════════════════════════════════════════════════
    #  RUN 1 — Full pipeline
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "─" * 64)
    print("  RUN 1 — Full pipeline (fresh)")
    print("─" * 64)
    t_run1 = time.time()
    report1, reducer = run_pipeline(ds)
    print(f"\n  Run 1 completed in {time.time() - t_run1:.1f}s")

    # ── Checkpoint 1: Embeddings ──
    print("\n── Checkpoint 1: Embedding coverage ──")
    embedded = []
    for s in ds:
        try:
            emb = s["embedding"]
        except (KeyError, AttributeError):
            emb = None
        if emb is not None:
            embedded.append(s)

    min_expected = N_SAMPLES - 2  # tolerate up to 2 corrupt/failing videos
    check(
        f"≥ {min_expected} of {len(ds)} samples embedded successfully",
        len(embedded) >= min_expected,
        f"only {len(embedded)} embedded",
    )

    # ── Checkpoint 2: Required sample fields ──
    print("\n── Checkpoint 2: Sample fields ──")
    missing = []
    for s in embedded:
        fname = os.path.basename(s.filepath)
        for field in REQUIRED_FIELDS:
            try:
                val = s[field]
            except (KeyError, AttributeError):
                val = None
            if val is None:
                missing.append(f"{fname}.{field}")
    check(
        f"All {len(REQUIRED_FIELDS)} required fields present on "
        f"{len(embedded)} embedded samples",
        len(missing) == 0,
        f"missing {len(missing)}: {missing[:5]}{'…' if len(missing) > 5 else ''}",
    )

    # ── Checkpoint 3: Coverage score ──
    print("\n── Checkpoint 3: Coverage score ──")
    score = report1.get("coverage_score")
    check(
        "coverage_score is a float",
        isinstance(score, (int, float)),
        f"type={type(score).__name__}",
    )
    check(
        "coverage_score ∈ [0, 1]",
        score is not None and 0.0 <= float(score) <= 1.0,
        f"value={score}",
    )

    # ── Checkpoint 4: Clusters ──
    print("\n── Checkpoint 4: Cluster formation ──")
    cluster_ids = set()
    for s in ds:
        try:
            cid = s["cluster_id"]
        except (KeyError, AttributeError):
            continue
        if cid is not None and int(cid) >= 0:
            cluster_ids.add(int(cid))
    check(
        "≥ 1 non-noise cluster formed",
        len(cluster_ids) >= 1,
        f"found {len(cluster_ids)} clusters",
    )

    # ── Checkpoint 5: HTML report export ──
    print("\n── Checkpoint 5: Report export ──")
    if os.path.exists(REPORT_PATH):
        os.remove(REPORT_PATH)

    try:
        summary = vcga.export_coverage_report(ds, report1, REPORT_PATH)
        export_ok = True
    except Exception as e:
        export_ok = False
        check("export_coverage_report did not raise", False, str(e))

    if export_ok:
        exists = os.path.isfile(REPORT_PATH)
        check("Report file exists on disk", exists)
        if exists:
            size = os.path.getsize(REPORT_PATH)
            check(
                "Report is > 1 KB",
                size > 1024,
                f"size={size} bytes",
            )
            with open(REPORT_PATH, "r", encoding="utf-8") as f:
                head = f.read(200)
            check(
                "Report is valid HTML (starts with <!doctype)",
                head.strip().lower().startswith("<!doctype"),
                f"starts with: {head[:50]!r}",
            )

    # ── Checkpoint 6: History after run 1 ──
    print("\n── Checkpoint 6: History (1 entry) ──")
    history = ds.info.get(vcga.HISTORY_INFO_KEY, []) or []
    check(
        "coverage_history has exactly 1 entry after run 1",
        len(history) == 1,
        f"got {len(history)}",
    )

    # ═══════════════════════════════════════════════════════════════
    #  RUN 2 — Re-run (embeddings cached, testing diff)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "─" * 64)
    print("  RUN 2 — Re-run (cache warm, testing historical diff)")
    print("─" * 64)
    t_run2 = time.time()
    report2, _ = run_pipeline(ds)
    print(f"\n  Run 2 completed in {time.time() - t_run2:.1f}s")

    # ── Checkpoint 7: History has 2 entries ──
    print("\n── Checkpoint 7: History (2 entries) ──")
    history = ds.info.get(vcga.HISTORY_INFO_KEY, []) or []
    check(
        "coverage_history has exactly 2 entries after run 2",
        len(history) == 2,
        f"got {len(history)}",
    )

    # ── Checkpoint 8: Diff structure ──
    print("\n── Checkpoint 8: Historical diff ──")
    if len(history) >= 2:
        diff = vcga.compute_coverage_diff(history[-1], history[-2])
        check("compute_coverage_diff returns non-None", diff is not None)

        if diff is not None:
            for key in [
                "coverage_delta_pct",
                "closed_gaps",
                "still_open_gaps",
                "newly_opened_gaps",
                "new_samples",
                "n_samples_prev",
                "n_samples_curr",
            ]:
                check(f"diff has '{key}'", key in diff)

            # Print the diff summary for human inspection
            print(f"\n  Diff summary:")
            delta = diff.get("coverage_delta_pct", 0)
            sign = "+" if delta >= 0 else ""
            print(
                f"    Coverage: "
                f"{diff['coverage_score_prev'] * 100:.1f}% → "
                f"{diff['coverage_score_curr'] * 100:.1f}% "
                f"({sign}{delta:.1f} pts)"
            )
            print(
                f"    Samples:  "
                f"{diff['n_samples_prev']} → {diff['n_samples_curr']} "
                f"({sign}{diff['new_samples']})"
            )
            if diff.get("closed_gaps"):
                print(f"    Closed:   {', '.join(diff['closed_gaps'])}")
            if diff.get("still_open_gaps"):
                print(f"    Open:     {', '.join(diff['still_open_gaps'])}")
            if diff.get("newly_opened_gaps"):
                print(f"    New gaps: {', '.join(diff['newly_opened_gaps'])}")
    else:
        check(
            "Skipping diff — not enough history entries",
            False,
            "need 2, got " + str(len(history)),
        )

    # ═══════════════════════════════════════════════════════════════
    #  Cleanup
    # ═══════════════════════════════════════════════════════════════
    print("\n── Cleanup ──")
    if os.path.exists(REPORT_PATH):
        os.remove(REPORT_PATH)
        print(f"  Removed {os.path.basename(REPORT_PATH)}")

    try:
        fo.delete_dataset(DATASET_NAME)
        print(f"  Deleted dataset '{DATASET_NAME}'")
    except Exception as ex:
        print(f"  ⚠  Could not delete dataset: {ex}")

    # ═══════════════════════════════════════════════════════════════
    #  Summary
    # ═══════════════════════════════════════════════════════════════
    total = _passed + _failed
    print("\n" + "=" * 64)
    if _failed == 0:
        print(f"  ALL {total} CHECKS PASSED ✅")
    else:
        print(f"  {_passed}/{total} passed · {_failed} FAILED ❌")
    print("=" * 64)

    sys.exit(1 if _failed else 0)


# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Interrupted.")
        sys.exit(130)
    except Exception:
        print("\n\n  💥 Unhandled exception:")
        traceback.print_exc()
        sys.exit(2)
