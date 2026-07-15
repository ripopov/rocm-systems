# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
CDNA Memory Architecture Diagram - CLI Visualization
=============================================================================
Memory chart renderer for CDNA-class GPUs (MI200, MI300, MI350 series).
For RDNA3.5 (gfx1151) see mem_chart_gfx11.py.

USAGE:
    python mem_chart_gfx9.py [--data metrics.json] [--debug]
        [--txt file.txt] [--svg file.svg]

API:
    normalize_mem_chart_metrics(metric_dict) -> flat ordered dict for UIs
    plot_mem_chart(..., *, chart_title=...) -> str
    format_mem_chart_heading(normal_unit, *, panel_id=300, section_label=...) -> str

Metric dict keys must match the Memory Chart panel YAML for CDNA:

    src/rocprof_compute_soc/analysis_configs/gfx9*/0300_memory_chart.yaml

Use ``MEM_CHART_PANEL_METRIC_KEYS`` for the authoritative ordered list.
"""

import argparse
import json
import pathlib
import re
from io import StringIO
from typing import Any, Optional, Union

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from utils.mem_chart_common import (
    COLORS,
    _fmt_edge,
    _safe_float_sum,
    bar,
    metric_line,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MEM_CHART_DEFAULT_ROWS: tuple[tuple[str, Union[int, float, None]], ...] = (
    ("Wavefront Occupancy", 8),
    ("Wave Life", 4200),
    ("SALU", 1200),
    ("SMEM", 45),
    ("VALU", 3500),
    ("Matrix Ops", 800),
    ("VMEM", 220),
    ("LDS", 150),
    ("GWS", 0),
    ("BR", 90),
    ("Active CUs (deprecated)", 110),
    ("Num CUs", 110),
    ("VGPR", 64),
    ("SGPR", 32),
    ("LDS Allocation", 32768),
    ("Scratch Allocation", 0),
    ("Wavefronts", 16384),
    ("Workgroups", 256),
    ("Flat Read", 80),
    ("Flat Write", 20),
    ("Flat Atomic", 4),
    ("Buffer Read", 3000),
    ("Buffer Write", 400),
    ("Buffer Atomic", 8),
    ("LDS Req", 150),
    ("LDS Util", 45),
    ("LDS Latency", 28),
    ("LDS Read", None),
    ("LDS Write", None),
    ("LDS Atomic", None),
    ("VL1 Rd", 3200),
    ("VL1 Wr", 480),
    ("VL1 Atomic", 12),
    ("VL1 Hit", 92),
    ("VL1 Lat", 180),
    ("VL1 Coalesce", 87),
    ("VL1 Stall", 5),
    ("VL1_L2 Rd", 256),
    ("VL1_L2 Wr", 48),
    ("VL1_L2 Atomic", 12),
    ("sL1D Rd", 45),
    ("sL1D Hit", 98),
    ("sL1D Lat", 85),
    ("sL1D_L2 Rd", 1),
    ("sL1D_L2 Wr", 0),
    ("sL1D_L2 Atomic", 0),
    ("IL1 Fetch", 32),
    ("IL1 Hit", 99),
    ("IL1 Lat", 42),
    ("IL1_L2 Rd", 1),
    ("L2 Rd", 300),
    ("L2 Wr", 52),
    ("L2 Atomic", 12),
    ("L2 Hit", 85),
    ("L2 Rd Lat", 220),
    ("L2 Wr Lat", 180),
    ("Fabric_L2 Rd", 45),
    ("Fabric_L2 Wr", 8),
    ("Fabric_L2 Atomic", 1),
    ("Fabric Rd Lat", 350),
    ("Fabric Wr Lat", 280),
    ("Fabric Atomic Lat", 310),
    ("HBM Rd", 42),
    ("HBM Wr", 7),
)

MEM_CHART_PANEL_METRIC_KEYS: tuple[str, ...] = tuple(
    k for k, _ in _MEM_CHART_DEFAULT_ROWS
)

DEFAULT_SAMPLE_METRICS: dict[str, Union[int, float, None]] = dict(
    _MEM_CHART_DEFAULT_ROWS
)


# ---------------------------------------------------------------------------
# Public API helpers
# ---------------------------------------------------------------------------


def normalize_mem_chart_metrics(
    metric_dict: dict[str, Any],
) -> dict[str, Any]:
    """Normalize input to flat ordered dict with all panel keys present."""
    return {k: metric_dict.get(k) for k in MEM_CHART_PANEL_METRIC_KEYS}


def format_mem_chart_heading(
    normal_unit: str,
    *,
    panel_id: int = 300,
    section_label: str = "Memory Chart",
) -> str:
    section_num = panel_id // 100
    return f"{section_num}. {section_label} (Normalization: {normal_unit})"


def get_sample_metrics() -> dict[str, Any]:
    return dict(_MEM_CHART_DEFAULT_ROWS)


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def _extract_metrics(metric_dict: dict[str, Any]) -> dict[str, Any]:
    """Pull all needed values from the flat metric dict.

    Keys that are absent from the metric dict return None, which the
    rendering functions use to decide what to show — no architecture
    string checks needed.
    """
    get = metric_dict.get
    m: dict[str, Any] = {}

    # Non-buffer (Flat) and Buffer request breakdown
    m["flat_read"] = get("Flat Read")
    m["flat_write"] = get("Flat Write")
    m["flat_atomic"] = get("Flat Atomic")
    m["buffer_read"] = get("Buffer Read")
    m["buffer_write"] = get("Buffer Write")
    m["buffer_atomic"] = get("Buffer Atomic")

    # LDS
    m["lds_req"] = get("LDS Req")
    m["lds_util"] = get("LDS Util")
    m["lds_lat"] = get("LDS Latency")
    m["lds_read"] = get("LDS Read")
    m["lds_write"] = get("LDS Write")
    m["lds_atomic"] = get("LDS Atomic")

    # Kernel→cache edges
    m["smem_rd"] = get("sL1D Rd")
    m["icache_rd"] = get("IL1 Fetch")

    # L1 cache internals
    m["vl1_hit"] = get("VL1 Hit")
    m["sl1d_hit"] = get("sL1D Hit")
    m["il1_hit"] = get("IL1 Hit")

    # L1→L2 edges
    m["vl1_l2_rd"] = get("VL1_L2 Rd")
    m["vl1_l2_wr"] = get("VL1_L2 Wr")
    m["vl1_l2_atomic"] = get("VL1_L2 Atomic")
    m["sl1d_l2_rd"] = get("sL1D_L2 Rd")
    m["il1_l2_rd"] = get("IL1_L2 Rd")

    # L2
    m["l2_hit"] = get("L2 Hit")
    m["l2_rd"] = get("L2 Rd")
    m["l2_wr"] = get("L2 Wr")
    m["l2_atomic"] = get("L2 Atomic")

    # L2→Fabric
    m["fabric_l2_rd"] = get("Fabric_L2 Rd")
    m["fabric_l2_wr"] = get("Fabric_L2 Wr")
    m["fabric_l2_atomic"] = get("Fabric_L2 Atomic")
    m["fabric_l2_wr_atomic"] = _safe_float_sum(
        get("Fabric_L2 Wr"), get("Fabric_L2 Atomic")
    )

    # Fabric latency
    m["fabric_rd_lat"] = get("Fabric Rd Lat")
    m["fabric_wr_lat"] = get("Fabric Wr Lat")
    m["fabric_atomic_lat"] = get("Fabric Atomic Lat")

    # HBM
    m["hbm_rd"] = get("HBM Rd")
    m["hbm_wr"] = get("HBM Wr")

    return m


# ---------------------------------------------------------------------------
# Diagram building
# ---------------------------------------------------------------------------


def _make_arrows(length: int = 8) -> dict[str, str]:
    return {
        "left": "<" + "─" * length,
        "right": "─" * length + ">",
        "both": "<" + "─" * (length - 1) + ">",
    }


def _build_kernel_panel() -> Panel:
    return Panel(
        "\n" * 6 + "[dim]Shader Core[/dim]\n[dim]Wave Execution[/dim]",
        title=f"[bold {COLORS['kernel']}]Kernel[/bold {COLORS['kernel']}]",
        border_style=COLORS["kernel"],
        width=14,
        height=30,
    )


def _build_request_edges(
    m: dict[str, Any],
    arrows: dict[str, str],
) -> Text:
    """Edges from Kernel to Requests column."""
    c_rd = COLORS["read"]
    c_wr = COLORS["write"]
    c_at = COLORS["atomic"]
    ka_l = arrows["left"]
    ka_r = arrows["right"]
    ka_b = arrows["both"]

    lines = [
        "[white]Non-buffer[/white]",
        f"[{c_rd}]{_fmt_edge('Read', m['flat_read'])}[/{c_rd}]",
        f"[{c_rd}]{ka_l}[/{c_rd}]",
        f"[{c_wr}]{_fmt_edge('Write', m['flat_write'])}[/{c_wr}]",
        f"[{c_wr}]{ka_r}[/{c_wr}]",
        f"[{c_at}]{_fmt_edge('Atomic', m['flat_atomic'])}[/{c_at}]",
        f"[{c_at}]{ka_b}[/{c_at}]",
        "[white]Buffer[/white]",
        f"[{c_rd}]{_fmt_edge('Read', m['buffer_read'])}[/{c_rd}]",
        f"[{c_rd}]{ka_l}[/{c_rd}]",
        f"[{c_wr}]{_fmt_edge('Write', m['buffer_write'])}[/{c_wr}]",
        f"[{c_wr}]{ka_r}[/{c_wr}]",
        f"[{c_at}]{_fmt_edge('Atomic', m['buffer_atomic'])}[/{c_at}]",
        f"[{c_at}]{ka_b}[/{c_at}]",
        "",
    ]

    # LDS edges — data-driven: Rd/Wr/Atomic when available, else single Instr
    if m["lds_read"] is not None:
        lines.append("[white]LDS[/white]")
        lines.append(f"[{c_rd}]{_fmt_edge('Read', m['lds_read'])}[/{c_rd}]")
        lines.append(f"[{c_rd}]{ka_l}[/{c_rd}]")
        lines.append(f"[{c_wr}]{_fmt_edge('Write', m['lds_write'])}[/{c_wr}]")
        lines.append(f"[{c_wr}]{ka_r}[/{c_wr}]")
        lines.append(f"[{c_at}]{_fmt_edge('Atomic', m['lds_atomic'])}[/{c_at}]")
        lines.append(f"[{c_at}]{ka_b}[/{c_at}]")
    else:
        lines.append("[white]LDS[/white]")
        lines.append(f"[{c_rd}]{_fmt_edge('Instr', m['lds_req'])}[/{c_rd}]")
        lines.append(f"[{c_rd}]{ka_b}[/{c_rd}]")
        lines.append("")
        lines.append("")
        lines.append("")
        lines.append("")

    lines.append("")
    lines.append("[white]SMEM[/white]")
    lines.append(f"[{c_rd}]{_fmt_edge('Read', m['smem_rd'])}[/{c_rd}]")
    lines.append(f"[{c_rd}]{ka_l}[/{c_rd}]")
    lines.append("[white]ICACHE[/white]")
    lines.append(f"[{c_rd}]{_fmt_edge('Read', m['icache_rd'])}[/{c_rd}]")
    lines.append(f"[{c_rd}]{ka_l}[/{c_rd}]")

    return Text.from_markup("\n".join(lines))


def _build_l1_stack(m: dict[str, Any]) -> Table:
    """Build vertically stacked L1 cache panels: VL1D, LDS, sL1D, L1I."""
    c_bl = COLORS["block"]

    vl1_panel = Panel(
        f"{metric_line('Hit', m['vl1_hit'], '%', COLORS['hit'])}\n"
        f"[dim]{bar(m['vl1_hit'])}[/dim]",
        title=f"[bold {c_bl}]VL1D[/bold {c_bl}]",
        border_style=c_bl,
        width=18,
        height=7,
    )

    lds_panel = Panel(
        f"{metric_line('Util', m['lds_util'], '%', COLORS['util'])}\n"
        f"[dim]{bar(m['lds_util'])}[/dim]\n"
        f"{metric_line('Lat', m['lds_lat'], ' cyc', COLORS['stall'])}",
        title=f"[bold {COLORS['lds']}]LDS[/bold {COLORS['lds']}]",
        border_style=COLORS["lds"],
        width=18,
        height=8,
    )

    sl1d_panel = Panel(
        f"{metric_line('Hit', m['sl1d_hit'], '%', COLORS['hit'])}\n"
        f"[dim]{bar(m['sl1d_hit'])}[/dim]",
        title=f"[bold {c_bl}]sL1D[/bold {c_bl}]",
        border_style=c_bl,
        width=18,
        height=5,
    )

    l1i_panel = Panel(
        f"{metric_line('Hit', m['il1_hit'], '%', COLORS['hit'])}\n"
        f"[dim]{bar(m['il1_hit'])}[/dim]",
        title=f"[bold {c_bl}]L1I[/bold {c_bl}]",
        border_style=c_bl,
        width=18,
        height=5,
    )

    stack = Table.grid(padding=0)
    stack.add_column()
    stack.add_row(vl1_panel)
    stack.add_row(lds_panel)
    stack.add_row(sl1d_panel)
    stack.add_row(l1i_panel)
    return stack


def _build_l1_l2_edges(
    m: dict[str, Any],
    arrows: dict[str, str],
) -> Text:
    """L1→L2 edge column: VL1D Rd/Wr/Atomic, sL1D Rd, L1I Rd."""
    c_rd = COLORS["read"]
    c_wr = COLORS["write"]
    c_at = COLORS["atomic"]
    sa_l = arrows["left"]
    sa_r = arrows["right"]

    lines = [
        "[white]L1-L2[/white]",
        f"[{c_rd}]{_fmt_edge('Read', m['vl1_l2_rd'])}[/{c_rd}]",
        f"[{c_rd}]{sa_l}[/{c_rd}]",
        f"[{c_wr}]{_fmt_edge('Write', m['vl1_l2_wr'])}[/{c_wr}]",
        f"[{c_wr}]{sa_r}[/{c_wr}]",
        f"[{c_at}]{_fmt_edge('Atomic', m['vl1_l2_atomic'])}[/{c_at}]",
        f"[{c_at}]{arrows['both']}[/{c_at}]",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "[white]sL1D-L2[/white]",
        f"[{c_rd}]{_fmt_edge('Read', m['sl1d_l2_rd'])}[/{c_rd}]",
        f"[{c_rd}]{sa_l}[/{c_rd}]",
        "[white]L1I-L2[/white]",
        f"[{c_rd}]{_fmt_edge('Read', m['il1_l2_rd'])}[/{c_rd}]",
        f"[{c_rd}]{sa_l}[/{c_rd}]",
    ]
    return Text.from_markup("\n".join(lines))


def _build_l2_panel(m: dict[str, Any]) -> Panel:
    c_bl = COLORS["block"]
    return Panel(
        f"{metric_line('Hit', m['l2_hit'], '%', COLORS['hit'])}\n"
        f"[dim]{bar(m['l2_hit'])}[/dim]\n"
        "\n"
        f"{metric_line('Rd', m['l2_rd'], '', COLORS['read'])}\n"
        f"{metric_line('Wr', m['l2_wr'], '', COLORS['write'])}\n"
        f"{metric_line('Atomic', m['l2_atomic'], '', COLORS['atomic'])}",
        title=f"[bold {c_bl}]L2[/bold {c_bl}]",
        border_style=c_bl,
        width=18,
        height=30,
    )


def _build_l2_fabric_edges(
    m: dict[str, Any],
    arrows: dict[str, str],
) -> Text:
    """L2↔Fabric edges: Read and Write/Atomic (combined per PNG)."""
    c_rd = COLORS["read"]
    c_wr = COLORS["write"]
    sa_l = arrows["left"]
    sa_r = arrows["right"]

    lines = [
        "",
        "",
        "",
        "",
        "",
        "",
        "[white]L2↔Fabric[/white]",
        f"[{c_rd}]{_fmt_edge('Read', m['fabric_l2_rd'])}[/{c_rd}]",
        f"[{c_rd}]{sa_l}[/{c_rd}]",
        f"[{c_wr}]{_fmt_edge('Wr/At', m['fabric_l2_wr_atomic'])}[/{c_wr}]",
        f"[{c_wr}]{sa_r}[/{c_wr}]",
        "",
    ]
    return Text.from_markup("\n".join(lines))


def _build_data_fabric_panel(m: dict[str, Any]) -> Panel:
    content = (
        "\n\n"
        f"[dim]Latency (cycles)[/dim]\n"
        f"  {metric_line('Rd', m['fabric_rd_lat'], ' cyc', COLORS['read'])}\n"
        f"  {metric_line('Wr', m['fabric_wr_lat'], ' cyc', COLORS['write'])}\n"
        f"  {metric_line('At', m['fabric_atomic_lat'], ' cyc', COLORS['atomic'])}"
    )
    return Panel(
        content,
        title="[bold bright_magenta]Data Fabric[/bold bright_magenta]",
        border_style="bright_magenta",
        width=22,
        height=30,
    )


def _build_fabric_mem_edges(
    m: dict[str, Any],
    arrows: dict[str, str],
) -> Text:
    """Fabric→MALL/HBM edges."""
    c_rd = COLORS["read"]
    c_wr = COLORS["write"]
    sa_l = arrows["left"]
    sa_r = arrows["right"]

    lines = [
        "",
        "",
        "",
        "",
        "",
        "",
        f"[{c_rd}]{_fmt_edge('Rd', m['hbm_rd'])}[/{c_rd}]",
        f"[{c_rd}]{sa_l}[/{c_rd}]",
        f"[{c_wr}]{_fmt_edge('Wr', m['hbm_wr'])}[/{c_wr}]",
        f"[{c_wr}]{sa_r}[/{c_wr}]",
        "",
    ]
    return Text.from_markup("\n".join(lines))


def _build_mall_panel(m: dict[str, Any]) -> Panel:
    c_bl = COLORS["block"]
    return Panel(
        "\n\n[dim]Rd/Wr/Atomic BW\nTo Device-Mem[/dim]",
        title=f"[bold {c_bl}]MALL[/bold {c_bl}]",
        border_style="indian_red",
        width=18,
        height=30,
    )


def _build_umc_panel() -> Panel:
    return Panel(
        "",
        title=f"[bold {COLORS['block']}]UMC[/bold {COLORS['block']}]",
        border_style=COLORS["block"],
        width=8,
        height=30,
    )


def _build_hbm_panel() -> Panel:
    return Panel(
        "\n\n[bold bright_green]HBM[/bold bright_green]",
        title=f"[bold {COLORS['block']}]HBM[/bold {COLORS['block']}]",
        border_style="bright_yellow",
        width=10,
        height=30,
    )


def _build_xgmi_row(console: Console) -> None:
    """Render the xGMI block above the main diagram."""
    xgmi_panel = Panel(
        "[dim]XGMI (i.e., to Peer GPU)[/dim]",
        border_style="bright_yellow",
        width=30,
        height=3,
    )
    xgmi_layout = Table.grid(padding=0)
    xgmi_layout.add_column(width=90)
    xgmi_layout.add_column()
    xgmi_layout.add_row("", xgmi_panel)
    console.print(xgmi_layout)

    arrow_lines = Text.from_markup(
        " " * 100
        + "[dim]|^              Rd/Wr/Atomic BW[/dim]\n"
        + " " * 100
        + "[dim]||              to Infinity Fabric[/dim]\n"
        + " " * 100
        + "[dim]||[/dim]"
    )
    console.print(arrow_lines)


def _build_pcie_row(console: Console) -> None:
    """Render the PCIe block below the main diagram."""
    arrow_lines = Text.from_markup(
        " " * 100
        + "[dim]||[/dim]\n"
        + " " * 100
        + "[dim]||              Rd/Wr/Atomic BW[/dim]\n"
        + " " * 100
        + "[dim]V|              to PCIe[/dim]"
    )
    console.print(arrow_lines)

    pcie_panel = Panel(
        "[dim]PCIe (i.e., to CPU or Non-XGMI connected GPU)[/dim]",
        border_style="dark_olive_green3",
        width=50,
        height=3,
    )
    pcie_layout = Table.grid(padding=0)
    pcie_layout.add_column(width=80)
    pcie_layout.add_column()
    pcie_layout.add_row("", pcie_panel)
    console.print(pcie_layout)


def _print_scope_bar(console: Console) -> None:
    gpu_label = "[dim]GPU (XCD)[/dim]"
    fabric_label = "[dim]Fabric / Memory[/dim]"
    console.print(
        f"|{'─' * 60} {gpu_label} {'─' * 20}|{'─' * 10} {fabric_label} {'─' * 10}|"
    )


# ---------------------------------------------------------------------------
# Main diagram assembly
# ---------------------------------------------------------------------------


def create_mem_chart_diagram(
    metric_dict: dict[str, Any],
    console: Console,
    show_debug: bool = False,
    chart_title: str = "",
) -> None:
    """Create the CDNA memory diagram matching the reference PNG layout."""
    m = _extract_metrics(metric_dict)
    arrows = _make_arrows()

    console.print()
    if chart_title:
        console.print(f"[bold]{chart_title}[/bold]")

    # xGMI block (above main diagram)
    _build_xgmi_row(console)
    console.print()
    _print_scope_bar(console)
    console.print()

    # Main diagram row
    kernel = _build_kernel_panel()
    req_edges = _build_request_edges(m, arrows)
    l1_stack = _build_l1_stack(m)
    l1_l2_edges = _build_l1_l2_edges(m, arrows)
    l2 = _build_l2_panel(m)
    l2_fab_edges = _build_l2_fabric_edges(m, arrows)
    fabric = _build_data_fabric_panel(m)
    fab_mem_edges = _build_fabric_mem_edges(m, arrows)
    mall = _build_mall_panel(m)
    umc = _build_umc_panel()
    hbm = _build_hbm_panel()

    main_layout = Table.grid(padding=0)
    for _ in range(11):
        main_layout.add_column()

    main_layout.add_row(
        kernel,
        req_edges,
        l1_stack,
        l1_l2_edges,
        l2,
        l2_fab_edges,
        fabric,
        fab_mem_edges,
        mall,
        umc,
        hbm,
    )

    console.print(main_layout)
    console.print()

    # PCIe block (below main diagram)
    _build_pcie_row(console)
    console.print()

    # Legend
    legend = (
        f"[dim]Legend:[/dim] "
        f"[{COLORS['read']}]<────[/{COLORS['read']}] Read  "
        f"[{COLORS['write']}]────>[/{COLORS['write']}] Write  "
        f"[{COLORS['atomic']}]<───>[/{COLORS['atomic']}] Atomic  "
        f"[{COLORS['util']}]█[/{COLORS['util']}] Util  "
        f"[{COLORS['hit']}]█[/{COLORS['hit']}] Hit%  "
        f"[{COLORS['stall']}]█[/{COLORS['stall']}] Stall"
    )
    console.print(legend)
    console.print()

    if show_debug:
        console.print("[dim]Architecture Notes (CDNA):[/dim]")
        console.print("  VL1D: Per-CU vector data cache (Buffer/Non-buffer requests)")
        console.print("  LDS: Local Data Share, on-CU scratchpad")
        console.print("  sL1D: Per-CU scalar data cache (SMEM requests)")
        console.print("  L1I: Per-CU instruction cache (ICACHE requests)")
        console.print("  L2 (TCC): Shared last-level cache")
        console.print("  Data Fabric: Infinity Fabric interconnect")
        console.print("  MALL: Mid-level Address Lookup Layer (MI300+)")
        console.print("  UMC: Unified Memory Controller")
        console.print("  HBM: High Bandwidth Memory")
        console.print(
            "  xGMI: Inter-GPU link (MI350 has individual counters;"
            " earlier cards use 'traffic to remote')"
        )
        console.print(
            "  PCIe: Host/non-xGMI link (MI350 has individual counters;"
            " earlier cards use 'traffic to remote')"
        )
        console.print()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def plot_mem_chart(
    normal_unit: str,
    metric_dict: dict[str, Any],
    *,
    chart_title: Optional[str] = None,
) -> str:
    """Plot the CDNA memory chart and return as string.

    ``chart_title``: full heading line; if omitted, uses
    ``format_mem_chart_heading`` with ``panel_id=300``.
    """
    flat = normalize_mem_chart_metrics(metric_dict)
    resolved_heading = (
        format_mem_chart_heading(normal_unit, panel_id=300)
        if chart_title is None
        else chart_title
    )
    buf = StringIO()
    console = Console(file=buf, force_terminal=True, width=240, height=80)
    create_mem_chart_diagram(
        flat,
        console,
        show_debug=False,
        chart_title=resolved_heading,
    )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _render_to_plain_text(
    metrics: dict[str, Any],
    heading: str,
    show_debug: bool,
) -> str:
    """Render chart to plain text (no ANSI escape codes)."""
    buf = StringIO()
    console = Console(
        file=buf,
        force_terminal=False,
        width=240,
        height=80,
    )
    create_mem_chart_diagram(
        metrics,
        console,
        show_debug=show_debug,
        chart_title=heading,
    )
    raw = buf.getvalue()
    return re.sub(r"\x1b\[[0-9;]*m", "", raw)


def main() -> None:
    arg_parser = argparse.ArgumentParser(
        description="CDNA Memory Chart - CLI",
    )
    arg_parser.add_argument(
        "--data",
        "-d",
        help="JSON file with metrics data",
    )
    arg_parser.add_argument(
        "--debug",
        action="store_true",
        help="Show debug info",
    )
    arg_parser.add_argument(
        "--norm",
        default="per_kernel",
        help="Normalization unit",
    )
    arg_parser.add_argument("--txt", help="Write plain text to file")
    arg_parser.add_argument("--svg", help="Write SVG to file")
    args = arg_parser.parse_args()

    if args.data:
        with pathlib.Path(args.data).open(encoding="utf-8") as f:
            metrics = json.load(f)
    else:
        metrics = get_sample_metrics()

    heading = format_mem_chart_heading(args.norm)

    if args.txt:
        clean = _render_to_plain_text(
            metrics,
            heading,
            args.debug,
        )
        with pathlib.Path(args.txt).open("w", encoding="utf-8") as f:
            f.write(clean)
        return

    if args.svg:
        console = Console(record=True, width=240, height=80)
        create_mem_chart_diagram(
            metrics,
            console,
            show_debug=args.debug,
            chart_title=heading,
        )
        console.save_svg(args.svg, title="CDNA Memory Chart")
        return

    console = Console(width=240)
    create_mem_chart_diagram(
        metrics,
        console,
        show_debug=args.debug,
        chart_title=heading,
    )


if __name__ == "__main__":
    main()
