# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Tests for HBM roofline benchmark improvements.

Tests the --roof-hbm-source dispatch logic, TransferBench output parsing,
and the improved HBM kernel source without requiring GPU hardware.
"""

import re
from collections import namedtuple
from unittest.mock import MagicMock, patch

import pytest
from common import SRC  # noqa: F401 (triggers sys.path setup)

PerfMetrics = namedtuple("PerfMetrics", ["mean", "low", "high"])

try:
    from argparser import omniarg

    _HAS_ARGPARSER = True
except ImportError:
    _HAS_ARGPARSER = False

try:
    from roofline.benchmark.gfx9 import benchmark_gfx9_base

    _HAS_BENCHMARK = True
except ImportError:
    _HAS_BENCHMARK = False

try:
    from roofline import run_benchmark as _run_benchmark_mod

    _HAS_RUN_BENCHMARK = True
except ImportError:
    _HAS_RUN_BENCHMARK = False


# =============================================================================
# TransferBench output parsing
# =============================================================================

TRANSFERBENCH_HBM_OUTPUT = """\
TransferBench v1.68.00 (develop:8999f31) (No extra feature support) (Single-node mode)
=============================================================================================================
Testing on at least 1073741824 bytes (24 configs per GPU): ..
\u250c------------\u252c--------------------------------------------\u2510
\u2502 Rank   GPU \u2502 MaxBw (GB/s)   AvgBw (GB/s)   MinBw (GB/s) \u2502
\u251c------------\u253c--------------------------------------------\u2524
\u2502    0     0 \u2502      3997.25        3939.56        3763.29 \u2502
\u2514------------\u2534--------------------------------------------\u2518
"""

TRANSFERBENCH_MULTI_GPU_OUTPUT = """\
TransferBench v1.68.00 (develop:8999f31) (No extra feature support) (Single-node mode)
=============================================================================================================
Testing on at least 1073741824 bytes (24 configs per GPU): ....
\u250c------------\u252c--------------------------------------------\u2510
\u2502 Rank   GPU \u2502 MaxBw (GB/s)   AvgBw (GB/s)   MinBw (GB/s) \u2502
\u251c------------\u253c--------------------------------------------\u2524
\u2502    0     0 \u2502      3997.25        3939.56        3763.29 \u2502
\u2502    1     1 \u2502      4010.50        3950.00        3800.00 \u2502
\u2514------------\u2534--------------------------------------------\u2518
"""


def test_parse_transferbench_single_gpu():
    """Verify regex extracts bandwidth values from TransferBench hbm output."""
    bw_values = re.findall(r"(\d+\.\d+)", TRANSFERBENCH_HBM_OUTPUT)
    assert len(bw_values) >= 3
    peak = max(float(v) for v in bw_values)
    assert peak == pytest.approx(3997.25)


def test_parse_transferbench_multi_gpu():
    """Verify regex picks the highest value across multiple GPUs."""
    bw_values = re.findall(r"(\d+\.\d+)", TRANSFERBENCH_MULTI_GPU_OUTPUT)
    peak = max(float(v) for v in bw_values)
    assert peak == pytest.approx(4010.50)


def test_parse_transferbench_no_match():
    """Verify empty output produces no matches."""
    bw_values = re.findall(r"(\d+\.\d+)", "No bandwidth data here")
    assert len(bw_values) == 0


def test_parse_transferbench_max_ignores_version():
    """max() correctly picks BW values over small version numbers like 1.68."""
    bw_values = re.findall(r"(\d+\.\d+)", TRANSFERBENCH_HBM_OUTPUT)
    peak = max(float(v) for v in bw_values)
    assert peak > 3000.0


# =============================================================================
# HBM benchmark dispatch (_select_hbm_benchmark)
# =============================================================================


class MockBenchGfx9:
    """Minimal mock of Bench_gfx9 to test _select_hbm_benchmark dispatch."""

    def __init__(self, hbm_source):
        self.hbm_source = hbm_source

    def _select_hbm_benchmark(self):
        if self.hbm_source == "transferbench":
            return self._transferbench
        if self.hbm_source == "auto":
            import shutil

            if shutil.which("TransferBench") is not None:
                return self._transferbench
        return self._builtin

    def _builtin(self, device):
        return PerfMetrics(2943.0, 2900.0, 2980.0)

    def _transferbench(self, device):
        return PerfMetrics(3987.0, 3787.0, 4186.0)


def test_dispatch_builtin():
    """--roof-hbm-source builtin always selects built-in kernel."""
    bench = MockBenchGfx9("builtin")
    func = bench._select_hbm_benchmark()
    assert func == bench._builtin


def test_dispatch_transferbench():
    """--roof-hbm-source transferbench always selects TransferBench."""
    bench = MockBenchGfx9("transferbench")
    func = bench._select_hbm_benchmark()
    assert func == bench._transferbench


def test_dispatch_auto_with_transferbench():
    """auto mode selects TransferBench when it's on $PATH."""
    bench = MockBenchGfx9("auto")
    with patch("shutil.which", return_value="/opt/rocm/bin/TransferBench"):
        func = bench._select_hbm_benchmark()
    assert func == bench._transferbench


def test_dispatch_auto_without_transferbench():
    """auto mode falls back to built-in when TransferBench is absent."""
    bench = MockBenchGfx9("auto")
    with patch("shutil.which", return_value=None):
        func = bench._select_hbm_benchmark()
    assert func == bench._builtin


# =============================================================================
# Kernel source validation (requires vendored pyyaml built)
# =============================================================================


@pytest.mark.skipif(not _HAS_BENCHMARK, reason="vendored deps not built")
def test_improved_kernel_uses_float4():
    """Verify the improved HBM kernel source uses float4 and iteration loop."""
    bench = MagicMock()
    benchmark_gfx9_base.Bench_gfx9._load_kernel_sources(bench)
    src = bench.hbm_bw_src
    assert "float4" in src
    assert "totalElements" in src
    assert "stride" in src
    assert 'extern "C"' in src
    assert "template" not in src
    assert "tid" not in src


# =============================================================================
# Argparser --roof-hbm-source flag (requires vendored pyyaml built)
# =============================================================================


@pytest.mark.skipif(not _HAS_ARGPARSER, reason="vendored deps not built")
def test_argparser_roof_hbm_source_default():
    """--roof-hbm-source defaults to 'auto'."""
    args = omniarg(["profile", "--roof-only", "--", "sleep", "1"])
    assert args.roof_hbm_source == "auto"


@pytest.mark.skipif(not _HAS_ARGPARSER, reason="vendored deps not built")
def test_argparser_roof_hbm_source_builtin():
    """--roof-hbm-source accepts 'builtin'."""
    args = omniarg([
        "profile",
        "--roof-only",
        "--roof-hbm-source",
        "builtin",
        "--",
        "sleep",
        "1",
    ])
    assert args.roof_hbm_source == "builtin"


@pytest.mark.skipif(not _HAS_ARGPARSER, reason="vendored deps not built")
def test_argparser_roof_hbm_source_transferbench():
    """--roof-hbm-source accepts 'transferbench'."""
    args = omniarg([
        "profile",
        "--roof-only",
        "--roof-hbm-source",
        "transferbench",
        "--",
        "sleep",
        "1",
    ])
    assert args.roof_hbm_source == "transferbench"


@pytest.mark.skipif(not _HAS_ARGPARSER, reason="vendored deps not built")
def test_argparser_roof_hbm_source_invalid():
    """--roof-hbm-source rejects invalid values."""
    with pytest.raises(SystemExit):
        omniarg([
            "profile",
            "--roof-only",
            "--roof-hbm-source",
            "invalid",
            "--",
            "sleep",
            "1",
        ])


# =============================================================================
# run_benchmark.py plumbing (requires vendored pyyaml built)
# =============================================================================


@pytest.mark.skipif(not _HAS_RUN_BENCHMARK, reason="vendored deps not built")
def test_load_bench_passes_hbm_source():
    """load_bench passes hbm_source to the bench class constructor."""
    mock_props = MagicMock()
    mock_props.gcnArchName = "gfx942:sramecc+:xnack-"

    mock_bench_class = MagicMock()
    mock_module = MagicMock()
    mock_module.Bench_gfx942 = mock_bench_class

    with (
        patch(
            "utils.hip_interface.hipGetDeviceProperties", return_value=mock_props
        ),
        patch("importlib.import_module", return_value=mock_module),
    ):
        _run_benchmark_mod.load_bench(0, {}, "transferbench")

    mock_bench_class.assert_called_once_with(0, {}, "transferbench")
