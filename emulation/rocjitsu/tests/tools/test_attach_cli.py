# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Check attach argument handling without a GPU or a running daemon."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    binary = str(Path(sys.argv[1]).resolve())
    with tempfile.TemporaryDirectory(prefix="rocjitsu-attach-cli-") as runtime:
        env = dict(os.environ, ROCJITSU_RUNTIME_DIR=runtime)

        def rejected(arguments, diagnostic):
            result = subprocess.run(
                [binary, *arguments], env=env, capture_output=True,
                text=True, timeout=10,
            )
            assert result.returncode != 0, result
            assert diagnostic in result.stderr, result.stderr

        # No config is needed to reach socket discovery. Do not manufacture a
        # device config or local simulator when the requested server is absent.
        rejected(["--attach", "--", "/bin/true"], "no daemon socket")
        rejected(["--", "/bin/true"], "--config is required")
        rejected(["--daemon", "--attach", "--", "/bin/true"],
                 "mutually exclusive")
        rejected(["--attach", "--config", str(Path(runtime) / "missing.json"),
                  "--", "/bin/true"], "config file not found")


if __name__ == "__main__":
    main()
