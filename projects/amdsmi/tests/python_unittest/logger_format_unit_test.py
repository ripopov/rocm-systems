#!/usr/bin/env python3
#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""Mock-based unit tests for ``AMDSMILogger`` human-readable list formatting.

These tests load ``amdsmi_logger`` directly from source with ``amdsmi_helpers``
stubbed, so they run without GPU hardware or the compiled ``amdsmi`` package.
They lock in the scalar-list rendering contract used by ``amd-smi static
--profile`` (and any other section that emits a list of plain strings):

* A scalar list item is indented strictly deeper than its own key, so items
  nest under the key instead of being outdented below a sibling field.
* Scalar list items are not prefixed with a ``- `` bullet.
* JSON output keeps the native list type (unchanged by the human-readable path).
"""

import importlib.util
import os
import sys
import types
import unittest


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
LOGGER_PATH = os.path.normpath(os.path.join(THIS_DIR, "..", "..", "amdsmi_cli", "amdsmi_logger.py"))


def _install_fake_helpers():
    """Register a stub ``amdsmi_helpers`` so ``amdsmi_logger`` imports cleanly.

    ``amdsmi_logger`` only needs the ``AMDSMIHelpers`` name at import time; the
    human-readable formatter under test never calls into it.
    """
    helpers = types.ModuleType("amdsmi_helpers")

    class AMDSMIHelpers:
        pass

    helpers.AMDSMIHelpers = AMDSMIHelpers
    sys.modules["amdsmi_helpers"] = helpers


def _load_logger_module():
    spec = importlib.util.spec_from_file_location("logger_under_test", LOGGER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _indent(line):
    return len(line) - len(line.lstrip(" "))


class TestHumanReadableScalarList(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _install_fake_helpers()
        cls.logger_mod = _load_logger_module()
        cls.logger = cls.logger_mod.AMDSMILogger(format="human_readable")

    def _render_profile(self, profiles):
        return self.logger._convert_json_to_human_readable(
            {"gpu": 0, "profile": {"available_profiles": profiles}}
        )

    def test_items_indented_deeper_than_key(self):
        """Each item must sit deeper than its AVAILABLE_PROFILES key (regression)."""
        profiles = ["CUSTOM", "VIDEO", "BOOTUP_DEFAULT"]
        lines = self._render_profile(profiles).splitlines()

        key_lines = [line for line in lines if line.strip() == "AVAILABLE_PROFILES:"]
        self.assertEqual(len(key_lines), 1, "expected exactly one AVAILABLE_PROFILES key line")
        key_indent = _indent(key_lines[0])

        for name in profiles:
            item_lines = [line for line in lines if line.strip() == name]
            self.assertEqual(len(item_lines), 1, f"expected one line for profile {name}")
            self.assertGreater(
                _indent(item_lines[0]),
                key_indent,
                f"profile {name} should be indented deeper than its key",
            )

    def test_no_dash_bullets(self):
        """Scalar list items must not render as ``- item`` bullets."""
        lines = self._render_profile(["CUSTOM", "VIDEO"]).splitlines()
        for line in lines:
            self.assertFalse(
                line.lstrip(" ").startswith("- "),
                f"unexpected dash bullet in human-readable output: {line!r}",
            )

    def test_items_do_not_outdent_between_sibling_fields(self):
        """Items nest under the list key while sibling scalars keep key-level indent.

        The original bug sandwiched the items at a shallower indent between
        AVAILABLE_PROFILES and the following CURRENT field.
        """
        out = self.logger._convert_json_to_human_readable(
            {
                "gpu": 0,
                "profile": {
                    "available_profiles": ["CUSTOM", "VIDEO"],
                    "current": "CUSTOM",
                    "num_profiles": 2,
                },
            }
        )
        lines = out.splitlines()
        key_indent = _indent(next(l for l in lines if l.strip() == "AVAILABLE_PROFILES:"))
        current_indent = _indent(next(l for l in lines if l.strip().startswith("CURRENT:")))
        self.assertEqual(
            current_indent, key_indent, "sibling field CURRENT should share the list key's indent"
        )
        for name in ("CUSTOM", "VIDEO"):
            item_indent = _indent(next(l for l in lines if l.strip() == name))
            self.assertGreater(item_indent, current_indent)


if __name__ == "__main__":
    unittest.main(verbosity=2)
