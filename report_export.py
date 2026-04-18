"""
report_export.py — Self-contained HTML export of a gap-analysis report.

Builds a single-file HTML document that viewers without FiftyOne installed
can open directly: the full three-tier report text formatted as styled HTML
tables, an inline SVG UMAP scatter plot rendered from the samples'
``umap_x`` / ``umap_y`` fields, and a base64-encoded CSV of per-cluster
statistics attached via a download link.

Public entry point: ``export_coverage_report(dataset, gap_report, output_path)``.
Everything else is internal markup building.
"""

from __future__ import annotations

import base64
import csv
import io
import logging
import os
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Optional, Union

from .gap_report import (
    HISTORY_INFO_KEY,
    build_coverage_history_entry,
    build_tiered_report,
    compute_coverage_diff,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Style + palette constants
# ------------------------------------------------------------

_CSS = """
:root {
  --fg: #1f2937;
  --muted: #6b7280;
  --border: #e5e7eb;
  --bg-soft: #f9fafb;
  --critical: #d32f2f;
  --moderate: #f57c00;
  --low: #9e9e9e;
  --quality-poor: #ef5350;
  --quality-fair: #ffb300;
  --quality-good: #26a69a;
  --quality-excellent: #43a047;
}
* { box-sizing: border-box; }
body {
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  color: var(--fg);
  line-height: 1.55;
  margin: 0;
  padding: 32px 48px 64px;
  background: white;
  max-width: 1100px;
}
header { margin-bottom: 32px; }
header h1 { margin: 0 0 6px; font-size: 28px; }
header .meta { color: var(--muted); margin: 0; font-size: 14px; }
h2 {
  margin-top: 36px;
  padding-bottom: 8px;
  border-bottom: 2px solid var(--border);
  font-size: 22px;
}
h3 { margin-top: 20px; font-size: 16px; }
section { margin-bottom: 28px; }
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
  margin-top: 10px;
}
th, td {
  padding: 8px 10px;
  border: 1px solid var(--border);
  text-align: left;
  vertical-align: top;
}
th {
  background: var(--bg-soft);
  font-weight: 600;
}
tr:nth-child(even) td { background: #fcfcfc; }
.banner {
  display: inline-flex;
  align-items: baseline;
  gap: 12px;
  padding: 14px 22px;
  border-radius: 8px;
  color: white;
  margin-bottom: 14px;
}
.banner-poor      { background: var(--quality-poor); }
.banner-fair      { background: var(--quality-fair); color: #2b2b2b; }
.banner-good      { background: var(--quality-good); }
.banner-excellent { background: var(--quality-excellent); }
.banner .pct   { font-size: 32px; font-weight: 700; }
.banner .label { font-size: 16px; text-transform: uppercase; letter-spacing: 0.08em; }
dl.stats { display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; margin: 8px 0 16px; font-size: 14px; }
dl.stats dt { color: var(--muted); }
dl.stats dd { margin: 0; font-weight: 500; }
.recommendation {
  padding: 10px 14px;
  background: var(--bg-soft);
  border-left: 4px solid var(--quality-good);
  border-radius: 3px;
  font-size: 14px;
}
.sev-critical td.severity,
tr.sev-critical td:nth-child(2) { color: var(--critical); font-weight: 600; }
.sev-moderate td.severity,
tr.sev-moderate td:nth-child(2) { color: var(--moderate); font-weight: 600; }
.sev-low td.severity,
tr.sev-low td:nth-child(2)      { color: var(--low); }
tr.status-sparse td:nth-child(3) { color: var(--critical); font-weight: 600; }
.download-link {
  display: inline-block;
  padding: 6px 14px;
  background: #1f2937;
  color: white;
  border-radius: 4px;
  text-decoration: none;
  font-size: 13px;
  margin: 6px 0 14px;
}
.download-link:hover { background: #0f172a; }
.diff {
  padding: 12px 18px;
  border-radius: 6px;
  margin: 0 0 24px;
  border-left: 5px solid var(--muted);
  background: #f4f6fa;
}
.diff.trend-up   { border-left-color: var(--quality-good); background: #e8f6f3; }
.diff.trend-down { border-left-color: var(--quality-poor); background: #fdecec; }
.diff h2 { border: none; padding: 0; margin: 0 0 8px; font-size: 16px; }
.diff .hero { font-size: 14px; margin: 0 0 6px; }
.diff ul { margin: 4px 0 0 0; padding-left: 20px; font-size: 13px; }
.diff li { margin: 2px 0; }
.diff .closed   { color: var(--quality-good); }
.diff .open     { color: var(--moderate); }
.diff .new-gap  { color: var(--critical); }
.umap-svg {
  display: block;
  max-width: 100%;
  height: auto;
  background: white;
  border: 1px solid var(--border);
  border-radius: 4px;
}
footer {
  margin-top: 48px;
  padding-top: 16px;
  border-top: 1px solid var(--border);
  color: var(--muted);
  font-size: 12px;
}
@media print {
  body { max-width: none; padding: 16px; }
  h2 { page-break-before: auto; }
  section { page-break-inside: avoid; }
}
"""

# tab10-like palette — distinct, print-safe hues
_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


# ------------------------------------------------------------
# UMAP scatter — inline SVG
# ------------------------------------------------------------

def _collect_sample_points(dataset) -> list:
    """Pull the minimum data per sample needed for the scatter plot."""
    points = []
    if dataset is None:
        return points
    for sample in dataset:
        try:
            ux = sample["umap_x"]
            uy = sample["umap_y"]
            cid = sample["cluster_id"]
            is_outlier = sample["is_outlier"]
        except (KeyError, AttributeError):
            continue
        if ux is None or uy is None or cid is None:
            continue
        points.append({
            "x": float(ux),
            "y": float(uy),
            "cluster_id": int(cid),
            "is_outlier": bool(is_outlier),
            "filename": os.path.basename(sample.filepath),
        })
    return points


def _build_umap_svg(
    points: list,
    gap_report: dict,
    width: int = 900,
    height: int = 520,
) -> str:
    """Render the UMAP projection as an inline SVG element."""
    if not points:
        return "<p><em>No UMAP coordinates available — run stage 2 first.</em></p>"

    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    pad_l, pad_r, pad_t, pad_b = 50, 180, 20, 48  # last is legend/title space
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    def sx(x: float) -> float:
        if x_max == x_min:
            return pad_l + plot_w / 2
        return pad_l + (x - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        # Flip: SVG y grows downward, UMAP y naturally goes up
        if y_max == y_min:
            return pad_t + plot_h / 2
        return pad_t + (1.0 - (y - y_min) / (y_max - y_min)) * plot_h

    cluster_ids = sorted({p["cluster_id"] for p in points if p["cluster_id"] >= 0})
    color_by_cid = {
        cid: _PALETTE[i % len(_PALETTE)] for i, cid in enumerate(cluster_ids)
    }
    has_noise = any(p["is_outlier"] for p in points)

    parts: list = []
    parts.append(
        f'<svg viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" class="umap-svg" role="img" '
        f'aria-label="UMAP projection scatter plot">'
    )

    # Plot frame
    parts.append(
        f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w:.0f}" height="{plot_h:.0f}" '
        f'fill="#fbfbfd" stroke="#d0d7de"/>'
    )

    # Data points
    for p in points:
        cx, cy = sx(p["x"]), sy(p["y"])
        title = f'{p["filename"]} — cluster {p["cluster_id"]}'
        if p["is_outlier"]:
            parts.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" '
                f'fill="none" stroke="#8a8a8a" stroke-width="1.4">'
                f'<title>{escape(title)} (outlier)</title></circle>'
            )
        else:
            color = color_by_cid.get(p["cluster_id"], "#666")
            parts.append(
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5.5" '
                f'fill="{color}" fill-opacity="0.78" stroke="#1f2937" stroke-width="0.4">'
                f'<title>{escape(title)}</title></circle>'
            )

    # Missing-category markers — hollow red diamonds
    for g in gap_report.get("category_gaps", []) or []:
        mx = g.get("umap_x")
        my = g.get("umap_y")
        if mx is None or my is None:
            continue
        cx, cy = sx(float(mx)), sy(float(my))
        pts_attr = f"{cx:.1f},{cy - 11:.1f} {cx + 11:.1f},{cy:.1f} {cx:.1f},{cy + 11:.1f} {cx - 11:.1f},{cy:.1f}"
        parts.append(
            f'<polygon points="{pts_attr}" fill="none" '
            f'stroke="#d32f2f" stroke-width="2">'
            f'<title>Missing: {escape(g.get("category", "?"))} '
            f'(similarity {g.get("similarity", 0):.2f})</title></polygon>'
        )

    # Axis labels
    parts.append(
        f'<text x="{pad_l + plot_w / 2:.0f}" y="{height - 12}" '
        f'font-size="13" text-anchor="middle" fill="#4b5563">UMAP-1</text>'
    )
    parts.append(
        f'<text x="16" y="{pad_t + plot_h / 2:.0f}" '
        f'font-size="13" text-anchor="middle" fill="#4b5563" '
        f'transform="rotate(-90, 16, {pad_t + plot_h / 2:.0f})">UMAP-2</text>'
    )

    # Legend
    leg_x = width - pad_r + 16
    leg_y = pad_t + 6
    parts.append(
        f'<text x="{leg_x}" y="{leg_y}" font-size="12" '
        f'font-weight="700" fill="#1f2937">Legend</text>'
    )
    row_h = 20
    cursor_y = leg_y + 14
    for cid in cluster_ids:
        color = color_by_cid[cid]
        parts.append(
            f'<circle cx="{leg_x + 8}" cy="{cursor_y}" r="5" '
            f'fill="{color}" fill-opacity="0.78" stroke="#1f2937" stroke-width="0.4"/>'
        )
        parts.append(
            f'<text x="{leg_x + 22}" y="{cursor_y + 4}" font-size="12" '
            f'fill="#1f2937">Cluster {cid}</text>'
        )
        cursor_y += row_h

    if has_noise:
        parts.append(
            f'<circle cx="{leg_x + 8}" cy="{cursor_y}" r="5" '
            f'fill="none" stroke="#8a8a8a" stroke-width="1.4"/>'
        )
        parts.append(
            f'<text x="{leg_x + 22}" y="{cursor_y + 4}" font-size="12" '
            f'fill="#1f2937">Outliers</text>'
        )
        cursor_y += row_h

    if gap_report.get("category_gaps"):
        diamond = (
            f"{leg_x + 8},{cursor_y - 5} "
            f"{leg_x + 13},{cursor_y} "
            f"{leg_x + 8},{cursor_y + 5} "
            f"{leg_x + 3},{cursor_y}"
        )
        parts.append(
            f'<polygon points="{diamond}" fill="none" '
            f'stroke="#d32f2f" stroke-width="1.5"/>'
        )
        parts.append(
            f'<text x="{leg_x + 22}" y="{cursor_y + 4}" font-size="12" '
            f'fill="#1f2937">Missing category</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


# ------------------------------------------------------------
# CSV
# ------------------------------------------------------------

def _build_cluster_csv(tiered: dict) -> str:
    """Render per-cluster statistics from Tier 3 as CSV text."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "cluster_id",
        "size",
        "status",
        "mean_centroid_distance",
        "max_centroid_distance",
        "cluster_label",
        "cluster_diversity",
    ])
    for c in tiered["tier3_full_breakdown"]["clusters"]:
        writer.writerow([
            c["cluster_id"],
            c["size"],
            c["status"],
            c["mean_centroid_distance"],
            c["max_centroid_distance"],
            c["cluster_label"] or "",
            c["cluster_diversity"] or "",
        ])
    return buf.getvalue()


def _csv_download_link(csv_text: str, filename: str = "cluster_stats.csv") -> str:
    """data-URI download link — keeps the HTML single-file."""
    b64 = base64.b64encode(csv_text.encode("utf-8")).decode("ascii")
    return (
        f'<a class="download-link" '
        f'href="data:text/csv;base64,{b64}" download="{escape(filename)}">'
        f"⬇ Download cluster stats (CSV)</a>"
    )


# ------------------------------------------------------------
# HTML rendering
# ------------------------------------------------------------

def _render_priority_table(t2_rows: list) -> str:
    if not t2_rows:
        return "<p><em>No gaps detected.</em></p>"
    rows_html = []
    for i, e in enumerate(t2_rows, start=1):
        sev_cls = f"sev-{e['severity'].lower()}"
        rows_html.append(
            f"<tr class='{sev_cls}'>"
            f"<td>{i}</td>"
            f"<td class='severity'>{escape(e['severity'])}</td>"
            f"<td>{e['priority_score']:.0f}/100</td>"
            f"<td>{escape(e['name'])}</td>"
            f"<td>{escape(e['closest_cluster'])}</td>"
            f"<td>{e['similarity']:.2f}</td>"
            f"<td>{escape(e['collection_recommendation'])}</td>"
            f"</tr>"
        )
    return (
        "<table class='gap-table'>"
        "<thead><tr>"
        "<th>#</th><th>Severity</th><th>Priority</th><th>Gap</th>"
        "<th>Closest Cluster</th><th>Similarity</th><th>Recommended action</th>"
        "</tr></thead>"
        "<tbody>" + "\n".join(rows_html) + "</tbody>"
        "</table>"
    )


def _render_cluster_table(clusters: list) -> str:
    if not clusters:
        return "<p><em>No per-cluster data.</em></p>"
    rows_html = []
    for c in clusters:
        status_cls = "status-sparse" if c["status"] == "sparse" else "status-ok"
        rows_html.append(
            f"<tr class='{status_cls}'>"
            f"<td>{c['cluster_id']}</td>"
            f"<td>{c['size']}</td>"
            f"<td>{c['status']}</td>"
            f"<td>{c['mean_centroid_distance']:.3f}</td>"
            f"<td>{c['max_centroid_distance']:.3f}</td>"
            f"<td>{escape(c['cluster_label'] or '')}</td>"
            f"<td>{escape(c['cluster_diversity'] or '')}</td>"
            f"</tr>"
        )
    return (
        "<table class='cluster-table'>"
        "<thead><tr>"
        "<th>Cluster</th><th>Size</th><th>Status</th>"
        "<th>Mean centroid dist</th><th>Max centroid dist</th>"
        "<th>Pegasus description</th><th>Diversity</th>"
        "</tr></thead>"
        "<tbody>" + "\n".join(rows_html) + "</tbody>"
        "</table>"
    )


def _render_wellcovered_table(rows: list) -> str:
    if not rows:
        return ""
    body = "\n".join(
        f"<tr><td>{escape(wc['parent'])}</td>"
        f"<td>{escape(wc['category'])}</td>"
        f"<td>{wc['similarity']:.2f}</td>"
        f"<td>{escape(wc['closest_cluster'])}</td></tr>"
        for wc in rows
    )
    return (
        "<h3>Well-covered categories (no action needed)</h3>"
        "<table class='wellcovered-table'>"
        "<thead><tr>"
        "<th>Parent</th><th>Category</th>"
        "<th>Similarity</th><th>Nearest cluster</th>"
        "</tr></thead>"
        "<tbody>" + body + "</tbody>"
        "</table>"
    )


def _render_diff_block(diff: Optional[dict]) -> str:
    """Render the "Since last run" callout when diff data is available."""
    if not diff:
        return ""

    delta_pct = diff["coverage_delta_pct"]
    delta_sign = "+" if delta_pct >= 0 else ""
    trend_cls = "trend-up" if delta_pct > 0 else "trend-down" if delta_pct < 0 else ""

    parts = [
        f"<section class='diff {trend_cls}'>",
        f"<h2>Since last run ({escape(diff.get('previous_timestamp', ''))})</h2>",
        (
            f"<p class='hero'><strong>Coverage:</strong> "
            f"{diff['coverage_score_prev']*100:.1f}% → "
            f"{diff['coverage_score_curr']*100:.1f}% "
            f"({delta_sign}{delta_pct:.1f} pts) &middot; "
            f"<strong>Samples:</strong> {diff['n_samples_prev']} → "
            f"{diff['n_samples_curr']} "
            f"({'+' if diff['new_samples'] >= 0 else ''}{diff['new_samples']})"
            f"</p>"
        ),
    ]
    bullets = []
    if diff["closed_gaps"]:
        bullets.append(
            f"<li class='closed'>✓ <strong>Closed ({len(diff['closed_gaps'])}):</strong> "
            + escape(", ".join(diff["closed_gaps"])) + "</li>"
        )
    if diff["still_open_gaps"]:
        bullets.append(
            f"<li class='open'>✗ <strong>Still open ({len(diff['still_open_gaps'])}):</strong> "
            + escape(", ".join(diff["still_open_gaps"])) + "</li>"
        )
    if diff["newly_opened_gaps"]:
        bullets.append(
            f"<li class='new-gap'>⚠ <strong>Newly opened ({len(diff['newly_opened_gaps'])}):</strong> "
            + escape(", ".join(diff["newly_opened_gaps"])) + "</li>"
        )
    if bullets:
        parts.append("<ul>" + "".join(bullets) + "</ul>")
    else:
        parts.append("<p><em>No gap changes since the last run.</em></p>")
    parts.append("</section>")
    return "\n".join(parts)


def _render_html(
    dataset_name: str,
    tiered: dict,
    svg_markup: str,
    csv_link: str,
    generated_at: str,
) -> str:
    """Assemble the final single-file HTML document."""
    t1 = tiered["tier1_executive"]
    t2 = tiered["tier2_priority_gaps"]
    t3 = tiered["tier3_full_breakdown"]

    banner_cls = f"banner-{t1['coverage_quality'].lower()}"
    diff_html = _render_diff_block(t1.get("diff"))

    outliers_html = (
        f"<p><strong>{t3['n_outliers']}</strong> outlier videos "
        f"(HDBSCAN noise — no cluster assignment).</p>"
        if t3.get("n_outliers") else ""
    )

    return (
        "<!doctype html>\n"
        "<html lang='en'>\n"
        "<head>\n"
        "<meta charset='utf-8'>\n"
        f"<title>Coverage Report — {escape(dataset_name)}</title>\n"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>\n"
        f"<style>{_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        "<header>\n"
        "<h1>Video Coverage Report</h1>\n"
        f"<p class='meta'>Dataset: <strong>{escape(dataset_name)}</strong> · "
        f"{t1['total_videos']} videos · {t1['total_clusters']} clusters · "
        f"generated {escape(generated_at)}</p>\n"
        "</header>\n"
        f"{diff_html}\n"
        "<section class='tier tier-1'>\n"
        "<h2>Tier 1 — Executive Summary</h2>\n"
        f"<div class='banner {banner_cls}'>"
        f"<span class='pct'>{t1['coverage_pct']:.1f}%</span>"
        f"<span class='label'>{escape(t1['coverage_quality'])}</span>"
        "</div>\n"
        "<dl class='stats'>\n"
        f"<dt>Videos analyzed</dt><dd>{t1['total_videos']}</dd>\n"
        f"<dt>Clusters found</dt><dd>{t1['total_clusters']}</dd>\n"
        f"<dt>Gaps detected</dt>"
        f"<dd>{t1['n_gaps_total']} "
        f"({t1['n_category_gaps']} category, {t1['n_sparse_clusters']} sparse)"
        "</dd>\n"
        "</dl>\n"
        f"<p class='recommendation'><strong>Recommended next step:</strong> "
        f"{escape(t1['recommendation'])}</p>\n"
        "</section>\n"
        "<section class='plot'>\n"
        "<h2>UMAP Projection</h2>\n"
        f"{svg_markup}\n"
        "</section>\n"
        "<section class='tier tier-2'>\n"
        "<h2>Tier 2 — Priority Gaps</h2>\n"
        f"{_render_priority_table(t2)}\n"
        "</section>\n"
        "<section class='tier tier-3'>\n"
        "<h2>Tier 3 — Full Breakdown</h2>\n"
        f"<p>{csv_link}</p>\n"
        "<h3>Per-cluster statistics</h3>\n"
        f"{_render_cluster_table(t3['clusters'])}\n"
        f"{outliers_html}\n"
        f"{_render_wellcovered_table(t3['well_covered_categories'])}\n"
        "</section>\n"
        "<footer>\n"
        "<p>Generated by the video-content-gap-analyzer FiftyOne plugin.</p>\n"
        "</footer>\n"
        "</body>\n"
        "</html>\n"
    )


# ------------------------------------------------------------
# Public entry point
# ------------------------------------------------------------

def export_coverage_report(
    dataset,
    gap_report: dict,
    output_path: Union[str, os.PathLike],
) -> dict:
    """Write a self-contained HTML report to ``output_path``.

    The HTML embeds:
      * the three-tier text report (headings + tables)
      * an inline SVG UMAP scatter plot built from per-sample umap_x/y
      * a base64-encoded CSV of per-cluster stats via a download link

    Returns a summary dict ``{output_path, size_bytes, n_samples, n_clusters}``.
    """
    output_path = Path(output_path).expanduser().resolve()

    # Compute the "since last run" diff the same way ShowGapReport does —
    # compare the live gap_report against the second-to-last history entry.
    current_entry = build_coverage_history_entry(gap_report, dataset)
    history = (
        (dataset.info.get(HISTORY_INFO_KEY, []) or [])
        if dataset is not None else []
    )
    previous_entry = history[-2] if len(history) >= 2 else None
    diff = compute_coverage_diff(current_entry, previous_entry)

    tiered = build_tiered_report(gap_report, dataset, diff=diff)
    points = _collect_sample_points(dataset)
    svg_markup = _build_umap_svg(points, gap_report)
    csv_text = _build_cluster_csv(tiered)
    csv_link = _csv_download_link(csv_text)

    dataset_name = getattr(dataset, "name", "unknown-dataset")
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_text = _render_html(
        dataset_name=dataset_name,
        tiered=tiered,
        svg_markup=svg_markup,
        csv_link=csv_link,
        generated_at=generated_at,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_text, encoding="utf-8")

    size_bytes = output_path.stat().st_size
    logger.info(
        "Exported coverage report: %s (%d bytes, %d samples)",
        output_path, size_bytes, len(points),
    )

    return {
        "output_path": str(output_path),
        "size_bytes": int(size_bytes),
        "n_samples": len(points),
        "n_clusters": len(tiered["tier3_full_breakdown"]["clusters"]),
        "csv_rows": len(tiered["tier3_full_breakdown"]["clusters"]),
    }
