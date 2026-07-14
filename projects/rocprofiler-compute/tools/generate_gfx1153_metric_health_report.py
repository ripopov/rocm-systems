#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate gfx1153 metric health HTML report from rocprof-compute analyze logs.

Analyze logs must be produced with ``--view table`` so chart-only panels (roofline,
memory charts, etc.) are emitted as plain metric tables. Default TTY output omits
metrics and yields incomplete coverage.

Designed for mega_kernel-only validation on RDNA 3.5 Krackan2 (gfx1153); additional
workloads can be passed on the CLI using name:path pairs.
"""

from __future__ import annotations

import argparse
import html
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from utils.utils_common import load_panel_configs  # noqa: E402

CONFIG_ARCH = "gfx115x"

ANSI = re.compile(r"\x1b\[[0-9;]*m")
ROW_RE = re.compile(
    r"^\s*[│|]\s*"
    r"([0-9]+(?:\.[0-9]+)*)\s*[│|]\s*"
    r"([^│|]+?)\s*[│|]\s*"
    r"(.+)$"
)
SUPPRESS_RE = re.compile(
    r"Not showing table with empty column\(s\):\s*(.+)$", re.IGNORECASE
)
LEVEL_COUNTER_RE = re.compile(r"_LEVEL(?:_sum)?|INFLIGHT_LEVEL", re.IGNORECASE)

COUNTER_SEMANTIC_OVERFLOW_NAMES = frozenset({
    "CP Utilization",
    "Pipeline Utilization - CMACC",
    "Pipeline Utilization - SMACC",
    "Pipeline Utilization - Double Precision",
    "Pipeline Utilization - XDL",
    "SPI Utilization",
})

ZERO_SUB_BUCKETS: tuple[str, ...] = (
    "zero_healthy",
    "zero_optional",
    "zero_sdk_bug",
    "zero_other",
)

HEALTHY_ZERO_NAME_FRAGMENTS: tuple[str, ...] = (
    "Alloc Failure",
    "FIFO Full",
    "Context Save",
    "Context Restore",
    "No-Allocation",
    "Throttle",
    "Backpressure",
    "Retry",
    "Timeout",
    "Error Rate",
    "Underflow",
    "Overflow Rate",
    "Conflict Rate",
    "Stall Rate",
)

OPTIONAL_ZERO_NAME_FRAGMENTS: tuple[str, ...] = (
    "FP64",
    "FP8",
    "FP6",
    "FP4",
    "Int4",
    "Sparse",
    "VALU FLOPs (F64)",
    "Double Precision",
)

SDK_BUG_ZERO_NAME_FRAGMENTS: tuple[str, ...] = (
    "VALU FLOPs",
    "VALU IOPs",
    "VALU Trans FLOPs",
    "WMMA FLOPs",
    "DRAM Bandwidth - Read",
    "GL2 to DRAM Read",
    "GL2 to DRAM Read Latency",
    "Buffer Atomics Instructions",
)

ZERO_BUCKET_LABELS: dict[str, str] = {
    "zero_healthy": "Healthy zero (0% stall/failure)",
    "zero_optional": "Optional path (not exercised)",
    "zero_sdk_bug": "SDK / counter bug suspect",
    "zero_other": "Other all-workload zero",
}

ZERO_BUCKET_CARD_STYLES: dict[str, tuple[str, str]] = {
    "zero_healthy": ("#e8f5e9", "#2e7d32"),
    "zero_optional": ("#e3f2fd", "#1565c0"),
    "zero_sdk_bug": ("#ffcdd2", "#c62828"),
    "zero_other": ("#f5f5f5", "#616161"),
}


@dataclass
class MetricDef:
    metric_id: str
    name: str
    unit: str
    formula: str
    panel_block: int
    uses_level_counter: bool


@dataclass
class MetricValue:
    avg: float | None
    raw: str
    cls: str = ""


def strip_ansi(text: str) -> str:
    return ANSI.sub("", text)


def to_float(value: str) -> float | None:
    cleaned = value.strip().replace(",", "")
    if not cleaned or cleaned.upper() in {"N/A", "NAN", "-", ""}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def uses_level_counter(formula: str) -> bool:
    return bool(LEVEL_COUNTER_RE.search(formula))


def classify_all_workload_zero(name: str, formula: str) -> str:
    if uses_level_counter(formula):
        return "zero_sdk_bug"
    if any(fragment in name for fragment in HEALTHY_ZERO_NAME_FRAGMENTS):
        return "zero_healthy"
    if any(fragment in name for fragment in SDK_BUG_ZERO_NAME_FRAGMENTS):
        return "zero_sdk_bug"
    if any(fragment in name for fragment in OPTIONAL_ZERO_NAME_FRAGMENTS):
        return "zero_optional"
    formula_upper = formula.upper()
    if any(token in formula_upper for token in ("FP64", "FP8", "FP6", "FP4", "SPARSE")):
        return "zero_optional"
    return "zero_other"


def parse_log(path: Path) -> tuple[dict[str, MetricValue], list[str]]:
    text = strip_ansi(path.read_text(errors="replace"))
    metrics: dict[str, MetricValue] = {}
    suppressed: list[str] = []
    avg_col: int | None = None

    for line in text.splitlines():
        sup = SUPPRESS_RE.search(line)
        if sup:
            suppressed.append(sup.group(1).strip())
            continue

        if "Metric_ID" in line and "Avg" in line:
            cols = [c.strip() for c in re.split(r"[│|]", line) if c.strip()]
            avg_col = None
            for i, col in enumerate(cols):
                if col == "Avg":
                    avg_col = i
                    break
            continue

        match = ROW_RE.match(line)
        if not match or avg_col is None:
            continue

        metric_id = match.group(1).strip()
        name = match.group(2).strip()
        cols = [c.strip() for c in re.split(r"[│|]", match.group(3)) if c.strip()]
        data_avg_col = max(0, avg_col - 2)
        if data_avg_col >= len(cols):
            continue

        raw_avg = cols[data_avg_col]
        key = f"{metric_id}|{name}"
        metrics[key] = MetricValue(avg=to_float(raw_avg), raw=raw_avg)

    return metrics, suppressed


def classify_metrics(
    workload_metrics: dict[str, dict[str, MetricValue]],
    metric_units: dict[str, str],
    metric_formulas: dict[str, str],
) -> None:
    mega = workload_metrics.get("mega_kernel", {})
    for wl_name, wl_metrics in workload_metrics.items():
        for key, mv in wl_metrics.items():
            unit = metric_units.get(key, "")
            name = key.split("|", 1)[1]
            formula = metric_formulas.get(key, "")
            if mv.avg is None:
                mega_mv = mega.get(key)
                if (
                    wl_name != "mega_kernel"
                    and mega_mv is not None
                    and mega_mv.avg is not None
                ):
                    mv.cls = "expected_na"
                else:
                    mv.cls = "na"
                continue
            if "percent" in unit.lower() and abs(mv.avg) > 100:
                _name = key.split("|", 1)[1]
                if _name in COUNTER_SEMANTIC_OVERFLOW_NAMES:
                    mv.cls = "formula_overflow"
                else:
                    mv.cls = "overflow"
                continue
            mega_mv = mega.get(key)
            mega_avg = mega_mv.avg if mega_mv else None
            if abs(mv.avg) < 1e-12:
                if (
                    wl_name != "mega_kernel"
                    and mega_avg is not None
                    and abs(mega_avg) > 1e-12
                ):
                    mv.cls = "workload_zero"
                elif all(
                    wl_metrics.get(key, MetricValue(None, "")).avg is not None
                    and abs(wl_metrics.get(key, MetricValue(0.0, "0")).avg or 0) < 1e-12
                    for wl_metrics in workload_metrics.values()
                ):
                    mv.cls = classify_all_workload_zero(name, formula)
                else:
                    mv.cls = "good"
                continue
            mv.cls = "good"


def fmt_value(mv: MetricValue | None, unit: str = "") -> str:
    if mv is None or mv.avg is None:
        return "N/A"
    if "percent" in unit.lower():
        return f"{mv.avg:.2f}".rstrip("0").rstrip(".")
    if abs(mv.avg) >= 1000:
        return f"{mv.avg:,.2f}".rstrip("0").rstrip(".")
    return f"{mv.avg:.2g}"


def cell_style(cls: str) -> str:
    styles = {
        "good": "background:#c8e6c9;color:#1b5e20;font-weight:bold",
        "overflow": "background:#c62828;color:#fff;font-weight:bold",
        "formula_overflow": "background:#ff8f00;color:#fff;font-weight:bold",
        "workload_zero": "background:#b3e5fc;color:#01579b",
        "expected_na": "background:#d1c4e9;color:#4527a0",
        "na": "background:#e1bee7;color:#6a1b9a",
    }
    if cls in ZERO_BUCKET_CARD_STYLES:
        bg, fg = ZERO_BUCKET_CARD_STYLES[cls]
        weight = "font-weight:bold" if cls == "zero_sdk_bug" else ""
        return f"background:{bg};color:{fg};{weight}"
    return styles.get(cls, "")


def render_zero_sub_bucket_lines(summary: dict[str, int]) -> str:
    zero_total = sum(summary.get(b, 0) for b in ZERO_SUB_BUCKETS)
    if zero_total == 0:
        return "<div style='color:#555'>⬜ Zero (all workloads): <b>0</b></div>"
    lines = [
        f"<div style='color:#555'>⬜ Zero (all workloads): <b>{zero_total}</b></div>",
    ]
    for bucket in ZERO_SUB_BUCKETS:
        count = summary.get(bucket, 0)
        if count == 0:
            continue
        bg, fg = ZERO_BUCKET_CARD_STYLES[bucket]
        label = ZERO_BUCKET_LABELS[bucket]
        lines.append(
            f"<div style='color:{fg};font-size:10px;padding-left:8px'>"
            f"▪ {html.escape(label)}: <b>{count}</b></div>"
        )
    return "".join(lines)


def build_report(
    workloads: dict[str, Path],
    formulas: dict[str, tuple[str, str, bool]],
    baseline_logs: dict[str, Path] | None,
    out_path: Path,
    report_date: str,
    baseline_label: str,
    host_label: str,
    branch_label: str,
) -> None:
    wl_names = list(workloads.keys())
    parsed: dict[str, dict[str, MetricValue]] = {}
    suppressed: dict[str, list[str]] = {}
    for wl, log_path in workloads.items():
        parsed[wl], suppressed[wl] = parse_log(log_path)

    metric_units = {k: formulas.get(k, ("", "", False))[1] for k in formulas}
    metric_formulas = {k: formulas.get(k, ("", "", False))[0] for k in formulas}
    classify_metrics(parsed, metric_units, metric_formulas)

    all_keys: list[str] = []
    seen: set[str] = set()
    for wl in wl_names:
        for key in sorted(parsed[wl], key=lambda k: [int(p) for p in k.split("|")[0].split(".")]):
            if key not in seen:
                seen.add(key)
                all_keys.append(key)

    summary: dict[str, dict[str, int]] = {
        wl: {
            "total": 0,
            "good": 0,
            "workload_zero": 0,
            "expected_na": 0,
            "na": 0,
            "overflow": 0,
            "formula_overflow": 0,
            **{bucket: 0 for bucket in ZERO_SUB_BUCKETS},
        }
        for wl in wl_names
    }
    for key in all_keys:
        for wl in wl_names:
            mv = parsed[wl].get(key)
            if mv is None:
                continue
            summary[wl]["total"] += 1
            if mv.cls in summary[wl]:
                summary[wl][mv.cls] += 1

    cards = []
    descs = {
        "mega_kernel": "All-ops (RDNA35 mega kernel)",
        "vcopy": "Memory-bound",
        "nbody": "Compute-bound",
        "wmma_gemm": "WMMA",
    }
    for wl in wl_names:
        s = summary[wl]
        cards.append(
            f"<div class='card'><h3>{html.escape(wl)}</h3>"
            f"<div class='desc'>{descs.get(wl, '')}</div>"
            f"<div>Total: <b>{s['total']}</b></div>"
            f"<div style='color:#1b5e20'>✅ Good: {s['good']}</div>"
            f"{render_zero_sub_bucket_lines(s)}"
            f"<div style='color:#01579b'>⚪ Expected zero: {s['workload_zero']}</div>"
            f"<div style='color:#4527a0'>🔹 Expected N/A: {s['expected_na']}</div>"
            f"<div style='color:#e65100'>⚠️ N/A: {s['na']}</div>"
            f"<div style='color:#ff8f00'>📐 &gt;100% formula: {s['formula_overflow']}</div>"
            f"<div style='color:#c62828'>🚨 &gt;100% hardware: {s['overflow']}</div></div>"
        )

    improve_rows = []
    if baseline_logs:
        base_parsed = {wl: parse_log(p)[0] for wl, p in baseline_logs.items()}
        for key in all_keys:
            mid, name = key.split("|", 1)
            for wl in wl_names:
                old = base_parsed.get(wl, {}).get(key)
                new = parsed.get(wl, {}).get(key)
                if not old or not new:
                    continue
                if old.cls in {"overflow", "formula_overflow"} and new.cls not in {
                    "overflow",
                    "formula_overflow",
                }:
                    improve_rows.append(
                        f"<tr><td>{html.escape(wl)}</td><td class='mid'>{html.escape(mid)}</td>"
                        f"<td>{html.escape(name)}</td>"
                        f"<td style='background:#ffcdd2;font-weight:bold'>{html.escape(old.raw)}</td>"
                        f"<td style='background:#c8e6c9;font-weight:bold'>{html.escape(new.raw)}</td>"
                        f"<td>{html.escape(formulas.get(key, ('', '', False))[1])}</td></tr>"
                    )

    sup_rows = []
    for wl in wl_names:
        for table in suppressed.get(wl, []):
            sup_rows.append(
                f"<tr><td>{html.escape(wl)}</td><td>{html.escape(table)}</td></tr>"
            )

    body_rows = []
    for key in all_keys:
        mid, name = key.split("|", 1)
        formula, unit, level = formulas.get(key, ("", "", False))
        row_cats: set[str] = set()
        wl_cells = []
        for wl in wl_names:
            mv = parsed[wl].get(key)
            cls = mv.cls if mv else "na"
            row_cats.add(cls)
            val = fmt_value(mv, unit)
            style = cell_style(cls)
            wl_cells.append(
                f"<td style='{style}' data-cls='{cls}'>{html.escape(val)}</td>"
            )

        problematic = row_cats & {
            "na",
            "overflow",
            "formula_overflow",
            *ZERO_SUB_BUCKETS,
        }
        if problematic:
            level_cell = (
                f"<b style='color:#6a1b9a'>Yes</b>"
                if level
                else "<span style='color:#546e7a'>No</span>"
            )
        else:
            level_cell = "—"

        formula_html = html.escape(f"value: {formula}") if formula else "—"
        body_rows.append(
            f"<tr data-cats='{' '.join(sorted(row_cats))}'>"
            f"<td class='mid'>{html.escape(mid)}</td>"
            f"<td class='mname'>{html.escape(name)}</td>"
            f"<td class='unit'>{html.escape(unit)}</td>"
            f"<td style='text-align:center;font-size:11px'>{level_cell}</td>"
            f"<td><div style='overflow-x:auto;white-space:nowrap;max-width:600px;font-size:10px;"
            f"font-family:monospace;background:#f0f4f8;color:#1a237e;padding:2px 5px;"
            f"border-radius:3px;border:1px solid #c5cae9'>{formula_html}</div></td>"
            + "".join(wl_cells)
            + "</tr>"
        )

    wl_headers = "".join(
        f"<th>{html.escape(wl)}<br><small>{descs.get(wl, '')}</small></th>"
        for wl in wl_names
    )

    improve_section = ""
    if improve_rows:
        improve_section = (
            f"<h2>✅ Improvements vs {html.escape(baseline_label)} "
            f"({len(improve_rows)} metrics fixed)</h2>"
            "<table class='improve-table'><thead><tr><th>Workload</th><th>ID</th>"
            "<th>Metric</th><th>Old</th><th>New</th><th>Unit</th></tr></thead><tbody>"
            + "".join(improve_rows)
            + "</tbody></table>"
        )

    doc = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>gfx1153 Krackan2 — Metric Health {html.escape(report_date)}</title>
<style>
body{{font-family:'Segoe UI',Arial,sans-serif;background:#eceff1;margin:0;padding:16px;font-size:13px}}
h1{{color:#0d47a1;margin-bottom:4px}}h2{{color:#263238;border-bottom:2px solid #1565c0;padding-bottom:4px;margin-top:22px}}
.cards{{display:flex;gap:12px;flex-wrap:wrap;margin:8px 0}}
.card{{background:#fff;border-radius:8px;padding:10px 16px;box-shadow:0 2px 6px rgba(0,0,0,.12);min-width:140px}}
.card h3{{margin:0 0 3px;font-size:13px;color:#555}}.desc{{font-size:10px;color:#888;margin-bottom:4px}}
.improve-table{{border-collapse:collapse;width:100%;background:#fff;margin-top:6px;font-size:12px}}
.improve-table th{{background:#1565c0;color:#fff;padding:5px 8px;text-align:left}}
.improve-table td{{padding:3px 8px;border-bottom:1px solid #eee}}
.bucket-semantics{{border-collapse:collapse;width:100%;max-width:920px;background:#fff;margin:10px 0 4px;font-size:12px;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.bucket-semantics th{{background:#eceff1;color:#37474f;padding:8px 10px;text-align:left;font-weight:600;position:static}}
.bucket-semantics td{{padding:7px 10px;border-bottom:1px solid #e0e0e0;vertical-align:top}}
.bucket-semantics tr:last-child td{{border-bottom:none}}
.bucket-semantics td:first-child{{font-weight:600;white-space:nowrap}}
.legend{{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0;user-select:none}}
.filter-note{{font-size:11px;color:#888;margin:-4px 0 4px}}
.tbl-wrap{{overflow-x:auto;max-height:65vh;overflow-y:auto}}
table{{border-collapse:collapse;width:100%;background:#fff;font-size:12px}}
th{{background:#37474f;color:#fff;padding:6px 8px;text-align:left;position:sticky;top:0;z-index:2;white-space:nowrap}}
td{{padding:3px 5px;border-bottom:1px solid #e0e0e0;vertical-align:middle}}
tr:hover td{{filter:brightness(0.91)}}
.mid{{font-family:monospace;color:#546e7a;font-size:11px}}.mname{{max-width:180px}}
.unit{{color:#90a4ae;font-style:italic;font-size:10px}}
tr.row-overflow{{box-shadow:inset 0 0 0 2px #c62828}}
.sup-table{{font-size:12px}}.sup-table td{{padding:3px 8px}}.sup-table tr:nth-child(even){{background:#f9f9f9}}
</style></head><body>
<h1>gfx1153 / RDNA35 Krackan2 — Metric Health Report</h1>
<p style="margin:2px 0;color:#607d8b;font-size:12px"><b>Date:</b> {html.escape(report_date)}
 | <b>Host:</b> {html.escape(host_label)}
 | <b>Branch:</b> {html.escape(branch_label)}
 | <b>Workload:</b> mega_kernel only</p>
<p style="font-size:11px;color:#555">Analysis configs: <code>{CONFIG_ARCH}</code>.
All-workload zeros are split into sub-buckets (healthy / optional path / SDK bug suspect / other).
With a single mega_kernel workload, Expected zero / Expected N/A buckets are typically empty.</p>
<h2>Summary</h2><div class="cards">{''.join(cards)}</div>
<h2>Bucket semantics</h2>
<p style="font-size:11px;color:#555;margin:2px 0 6px">Each metric in each workload column is assigned to <b>exactly one</b> bucket; counts on each card sum to Total.</p>
<table class="bucket-semantics">
<thead><tr><th>Bucket</th><th>Meaning</th><th>Healthy?</th></tr></thead>
<tbody>
<tr><td>Good</td><td>Parsed value exists; not N/A; not &gt;100% (percent); not classified as zero</td><td>Usually yes</td></tr>
<tr><td>Zero — healthy (<code>zero_healthy</code>)</td><td>Zero in every workload; metric is a stall/failure/alloc-failure rate where 0% is good</td><td>Yes</td></tr>
<tr><td>Zero — optional path (<code>zero_optional</code>)</td><td>Zero in every workload; precision or IP path mega_kernel does not exercise (FP64, etc.)</td><td>Usually yes</td></tr>
<tr><td>Zero — SDK bug suspect (<code>zero_sdk_bug</code>)</td><td>Zero in every workload but mega_kernel exercises the path or counters should fire (VALU FLOPs, DRAM read)</td><td>Investigate — file SDK JIRA</td></tr>
<tr><td>Zero — other (<code>zero_other</code>)</td><td>Zero in every workload; uncategorized (often low-traffic structural zeros)</td><td>Review case-by-case</td></tr>
<tr><td>Expected zero (<code>workload_zero</code>)</td><td>Zero here, but non-zero in mega_kernel</td><td>Yes — workload doesn't hit that path</td></tr>
<tr><td>Expected N/A (<code>expected_na</code>)</td><td>N/A here (formula guard / zero denominator), but mega_kernel has a value</td><td>Yes — workload doesn't hit that path</td></tr>
<tr><td>N/A</td><td>Analyze couldn't produce a value and mega_kernel also has no value</td><td>Investigate</td></tr>
<tr><td>&gt;100% formula (<code>formula_overflow</code>)</td><td>Percent &gt;100% from known counter semantics (OR-gate overlap, pipeline slot counting)</td><td>Usually benign — read trend not absolute %</td></tr>
<tr><td>&gt;100% hardware (<code>overflow</code>)</td><td>Percent &gt;100% not explained by known semantics — likely bad formula or counter bug</td><td>Investigate</td></tr>
</tbody></table>
{improve_section}
<h2>Suppressed Tables</h2>
<table class="sup-table"><tr style="background:#455a64;color:#fff"><th>Workload</th><th>Table</th></tr>
{''.join(sup_rows) if sup_rows else '<tr><td colspan=2>None</td></tr>'}
</table>
<h2>All Metrics — Blocks 2–18</h2>
<p style="font-size:11px;color:#555;margin:4px 0">Blocks 3 (Memory Chart) and 4 (Roofline) are chart/roofline panels — <b>no tabular metric rows in log output</b>.</p>
<div class="legend">
<div class="leg" data-cls="good" style="background:#c8e6c9;color:#1b5e20;font-weight:bold;padding:5px 12px;border-radius:5px;cursor:pointer;font-size:12px">▲ Good (real data)</div>
<div class="leg" data-cls="overflow" style="background:#c62828;color:#fff;font-weight:bold;padding:5px 12px;border-radius:5px;cursor:pointer;font-size:12px">🚨 &gt;100% hardware</div>
<div class="leg" data-cls="formula_overflow" style="background:#ff8f00;color:#fff;font-weight:bold;padding:5px 12px;border-radius:5px;cursor:pointer;font-size:12px">📐 &gt;100% formula</div>
<div class="leg" data-cls="zero_healthy" style="background:#e8f5e9;color:#2e7d32;padding:5px 12px;border-radius:5px;cursor:pointer;font-size:12px">✓ Zero — healthy (0% stall)</div>
<div class="leg" data-cls="zero_optional" style="background:#e3f2fd;color:#1565c0;padding:5px 12px;border-radius:5px;cursor:pointer;font-size:12px">○ Zero — optional path</div>
<div class="leg" data-cls="zero_sdk_bug" style="background:#ffcdd2;color:#c62828;font-weight:bold;padding:5px 12px;border-radius:5px;cursor:pointer;font-size:12px">❌ Zero — SDK bug suspect</div>
<div class="leg" data-cls="zero_other" style="background:#f5f5f5;color:#616161;padding:5px 12px;border-radius:5px;cursor:pointer;font-size:12px">⬜ Zero — other</div>
<div class="leg" data-cls="workload_zero" style="background:#b3e5fc;color:#01579b;padding:5px 12px;border-radius:5px;cursor:pointer;font-size:12px">⚪ Expected zero (non-zero in mega)</div>
<div class="leg" data-cls="expected_na" style="background:#d1c4e9;color:#4527a0;padding:5px 12px;border-radius:5px;cursor:pointer;font-size:12px">🔹 Expected N/A (valid in mega)</div>
<div class="leg" data-cls="na" style="background:#e1bee7;color:#6a1b9a;padding:5px 12px;border-radius:5px;cursor:pointer;font-size:12px">⚠️ N/A (investigate)</div>
</div>
<p class="filter-note">💡 Click legend to show/hide rows by category</p>
<div class="tbl-wrap"><table id="mt">
<thead><tr><th>ID</th><th>Metric Name</th><th>Unit</th><th>LEVEL<br><small>if problematic</small></th><th>Formula (scroll →)</th>{wl_headers}</tr></thead>
<tbody>{''.join(body_rows)}</tbody></table></div>
<script>
const active=new Set(['good','overflow','formula_overflow','zero_healthy','zero_optional','zero_sdk_bug','zero_other','workload_zero','expected_na','na']);
document.querySelectorAll('.leg').forEach(el=>{{
  el.addEventListener('click',()=>{{
    const c=el.dataset.cls;
    if(active.has(c)) active.delete(c); else active.add(c);
    el.style.opacity=active.has(c)?'1':'0.35';
    document.querySelectorAll('#mt tbody tr').forEach(row=>{{
      const cats=(row.dataset.cats||'').split(/\\s+/);
      row.style.display=cats.some(x=>active.has(x))?'':'none';
    }});
  }});
}});
</script></body></html>"""

    out_path.write_text(doc, encoding="utf-8")


def merge_formulas_from_logs(
    workloads: dict[str, Path],
) -> dict[str, tuple[str, str, bool]]:
    formulas: dict[str, tuple[str, str, bool]] = {}
    config_dir = SRC / "rocprof_compute_soc" / "analysis_configs" / CONFIG_ARCH
    yaml_formulas: dict[str, str] = {}
    yaml_units: dict[str, str] = {}

    for yaml_path in sorted(config_dir.glob("*.yaml")):
        data = yaml.safe_load(yaml_path.read_text())
        panel = data.get("Panel Config", data)
        for entry in panel.get("data source", []):
            if not isinstance(entry, dict):
                continue
            table = entry.get("metric_table", entry)
            if not isinstance(table, dict) or "metric" not in table:
                continue
            for name, spec in (table.get("metric") or {}).items():
                if not isinstance(spec, dict):
                    continue
                formula = str(spec.get("value", spec.get("avg", "")))
                unit = str(spec.get("unit", ""))
                yaml_formulas[name] = formula
                yaml_units[name] = unit

    for log_path in workloads.values():
        parsed, _ = parse_log(log_path)
        for key in parsed:
            _mid, name = key.split("|", 1)
            formula = yaml_formulas.get(name, "")
            unit = yaml_units.get(name, "")
            formulas[key] = (formula, unit, uses_level_counter(formula))

    return formulas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--baseline-label", default="none")
    parser.add_argument("--host", default="gorgon-point-1")
    parser.add_argument("--branch", default="users/feizheng10/gfx1153-enable")
    parser.add_argument("workload_log", nargs="+", help="name:path pairs")
    parser.add_argument(
        "--baseline",
        nargs="*",
        default=[],
        help="baseline workload logs as name:path",
    )
    args = parser.parse_args()

    workloads: dict[str, Path] = {}
    for spec in args.workload_log:
        name, path = spec.split(":", 1)
        workloads[name] = Path(path)

    baseline_logs: dict[str, Path] | None = None
    if args.baseline:
        baseline_logs = {}
        for spec in args.baseline:
            name, path = spec.split(":", 1)
            baseline_logs[name] = Path(path)

    formulas = merge_formulas_from_logs(workloads)
    build_report(
        workloads,
        formulas,
        baseline_logs,
        args.out,
        args.date,
        args.baseline_label,
        args.host,
        args.branch,
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
