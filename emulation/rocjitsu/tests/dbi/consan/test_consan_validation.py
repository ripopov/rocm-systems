#!/usr/bin/env python3

from __future__ import annotations

import json
import os
from pathlib import Path
import unittest
from unittest import mock

import consan_validation as validation
from consan_validation_test_support import temporary_root


class ConSanValidationTest(unittest.TestCase):
    def test_manifest_is_the_complete_north_star_matrix(self) -> None:
        manifest = validation._manifest("gfx1201")
        self.assertEqual(len(manifest["workloads"]), 11)
        self.assertEqual(
            [profile["id"] for profile in manifest["profiles"]],
            list(validation.PROFILE_IDS),
        )
        self.assertEqual(
            len({workload["id"] for workload in manifest["workloads"]}), 11
        )

    def test_profile_environment_scrubs_controls_and_relies_on_sync_defaults(self) -> None:
        workload = validation.WORKLOAD_BY_ID["streamk-arrival"]
        with mock.patch.dict(
            os.environ,
            {
                "RJ_CONSAN_MAX_PATCHES": "1",
                "RJ_CONSAN_TMP_VGPR": "99",
                "HSA_TOOLS_LIB": "/stale/hook.so",
            },
            clear=False,
        ):
            environment = validation._clean_environment(
                "record-replay", workload, Path("/new/hook.so")
            )
        self.assertNotIn("RJ_CONSAN_MAX_PATCHES", environment)
        self.assertNotIn("RJ_CONSAN_TMP_VGPR", environment)
        self.assertEqual(environment["HSA_TOOLS_LIB"], "/new/hook.so")
        self.assertNotIn("RJ_CONSAN_MOI_TRACK_BARRIERS", environment)
        self.assertNotIn("RJ_CONSAN_MOI_TRACK_ATOMICS", environment)
        self.assertEqual(
            validation.ORDINARY_MOI_RUNTIME_DEFAULTS,
            {
                "RJ_CONSAN_MOI_TRACK_BARRIERS": "1",
                "RJ_CONSAN_MOI_TRACK_ATOMICS": "1",
            },
        )

    def test_supercollider_does_not_receive_moi_tracking_controls(self) -> None:
        workload = validation.WORKLOAD_BY_ID["streamk-arrival"]
        environment = validation._clean_environment(
            "supercollider", workload, Path("/hook.so")
        )
        self.assertNotIn("RJ_CONSAN_MOI_TRACK_BARRIERS", environment)
        self.assertNotIn("RJ_CONSAN_MOI_TRACK_ATOMICS", environment)

    def test_qwen_sampled_relies_on_the_standard_runtime_operating_point(self) -> None:
        qwen = validation.WORKLOAD_BY_ID["qwen-prefill"]
        tp1 = validation.WORKLOAD_BY_ID["tp1-prefill"]
        qwen_environment = validation._clean_environment(
            "sampled", qwen, Path("/hook.so")
        )
        tp1_environment = validation._clean_environment(
            "sampled", tp1, Path("/hook.so")
        )
        self.assertEqual(qwen_environment["RJ_CONSAN_MOI_REQUIRE_RECORDS"], "1")
        self.assertNotIn("RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE", qwen_environment)
        self.assertNotIn("RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET", qwen_environment)
        self.assertNotIn("RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE", tp1_environment)

    def test_explain_expands_commands_and_marks_only_real_tuning(self) -> None:
        audit = validation._explain_contract(
            Path("/workspace"),
            "gfx1201",
            ("qwen-prefill",),
            validation.PROFILE_IDS,
            None,
            allow_reference=False,
        )
        workload = audit["workloads"][0]
        expected = validation._workload_command(
            Path("/workspace"),
            "gfx1201",
            validation.WORKLOAD_BY_ID["qwen-prefill"],
            "clean",
            Path("$ARTIFACT_ROOT/qwen-prefill/clean/$PROFILE/benchmark-0.json"),
        )
        self.assertEqual(workload["commands"]["clean"]["payload_argv"], expected)
        settings = {
            item["name"]: item
            for profile in workload["profiles"]
            for item in profile["settings"]
        }
        self.assertNotIn("CTEST_PARALLEL_LEVEL", settings)
        for forbidden in validation.ORDINARY_FORBIDDEN_ENVIRONMENT:
            self.assertNotIn(forbidden, settings)
        sampled = next(
            profile for profile in workload["profiles"] if profile["id"] == "sampled"
        )
        self.assertEqual(
            {
                item["name"]
                for item in sampled["implicit_runtime_defaults"]
            },
            {
                "RJ_CONSAN_MOI_TRACK_BARRIERS",
                "RJ_CONSAN_MOI_TRACK_ATOMICS",
                "RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET",
                "RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE",
            },
        )
        self.assertEqual(sampled["usability_exceptions"], [])
        self.assertEqual(
            audit["usability_audit"]["coverage_limiting_controls_present"], []
        )
        self.assertEqual(
            audit["usability_audit"]["explicit_event_family_overrides"], []
        )
        self.assertEqual(audit["usability_audit"]["workload_specific_tuning"], [])
        sampled_defaults = next(
            item
            for item in audit["usability_audit"]["automatic_profile_defaults"]
            if item["profile"] == "sampled"
        )
        self.assertEqual(
            sampled_defaults["settings"]["RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE"],
            "16384",
        )

    def test_explain_audits_reference_fault_outcomes_and_trial_knobs(self) -> None:
        path = Path(__file__).with_name(
            "consan_validation_faults_gfx1201_reference.json"
        )
        audit = validation._explain_contract(
            Path("/workspace"),
            "gfx1201",
            ("qwen-prefill",),
            validation.PROFILE_IDS,
            path,
            allow_reference=True,
        )
        fault = audit["workloads"][0]["faults"][0]
        sampled = next(
            item
            for item in fault["profile_expectations"]
            if item["profile"] == "sampled"
        )
        self.assertEqual(sampled["detector"], "statistical")
        self.assertEqual(sampled["oracle"], "fail")
        self.assertEqual(sampled["trial_count"], 32)
        self.assertEqual(
            sampled["trials"][0]["overrides"][0]["name"],
            "RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET",
        )
        self.assertIn("at least 1", sampled["required_diagnostic"])
        inline = next(
            item
            for item in fault["profile_expectations"]
            if item["profile"] == "inline-shadow"
        )
        effective = {
            setting["name"]: setting["value"]
            for setting in inline["trials"][0]["effective_settings"]
        }
        implicit = {
            setting["name"]: setting["value"]
            for setting in inline["trials"][0]["implicit_runtime_defaults"]
        }
        self.assertEqual(effective["RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS"], "1")
        self.assertNotIn("RJ_CONSAN_MOI_FORBID_DIAGNOSTICS", effective)
        self.assertEqual(implicit, validation.ORDINARY_MOI_RUNTIME_DEFAULTS)
        self.assertIn("$FAULT_SPEC", fault["validator_argv_template"])

    def test_tp1_decode_row_does_not_repeat_prefill(self) -> None:
        workload = validation.WORKLOAD_BY_ID["tp1-decode-combined"]
        command = validation._workload_command(
            Path("/workspace"), "gfx1201", workload, "clean", Path("/unused")
        )
        self.assertEqual(command[command.index("--mode") + 1], "decode-combined")

    def test_overhead_uses_bracketing_baseline_mean_and_maximum_mode(self) -> None:
        results = [
            {"profile": "baseline", "timing_median_ms": {"a": 2.0, "b": 4.0}},
            {"profile": "sampled", "timing_median_ms": {"a": 6.0, "b": 10.0}},
            {"profile": "baseline", "timing_median_ms": {"a": 4.0, "b": 6.0}},
        ]
        summary = validation._overhead_summary(results)
        self.assertEqual(summary["paired_baseline_median_ms"], {"a": 3.0, "b": 5.0})
        self.assertEqual(summary["profiles"]["sampled"]["cell_slowdown"], 2.0)

    def test_inventory_parser_deduplicates_exact_identities(self) -> None:
        output = "\n".join(
            (
                "ConSan fault site reader=1 identity=site-a kind=barrier "
                "sync_sequence=sequence-b",
                "ConSan fault site reader=2 identity=site-a kind=barrier",
                "ConSan sync sequence reader=1 identity=sequence-a kind=barrier",
                "ConSan barrier destination reader=1 identity=destination-a container=k",
            )
        )
        self.assertEqual(
            validation._inventory_records(output),
            {
                "sites": ["site-a"],
                "sequences": ["sequence-a", "sequence-b"],
                "destinations": ["destination-a"],
            },
        )

    def test_fault_inventory_enables_family_analysis_without_a_selector(self) -> None:
        barrier = validation._fault_inventory_environment("barrier-drop")
        self.assertEqual(barrier, {"RJ_CONSAN_FAULT_DROP_BARRIER": "1"})
        atomic = validation._fault_inventory_environment("atomic-weaken-order")
        self.assertEqual(
            atomic,
            {
                "RJ_CONSAN_FAULT_ATOMIC_WEAKEN_ORDER": "1",
                "RJ_CONSAN_FAULT_ATOMIC_ORDER_EDGE": "release",
            },
        )
        self.assertFalse(any(name.endswith("_IDENTITY") for name in atomic))

    def test_fault_acceptance_rejects_an_unattributed_process_signal(self) -> None:
        accepted, reasons = validation._fault_acceptance(
            {
                "mutation": {"requested": 1, "planned": 1, "applied": 1},
                "sanitizer": {"outcome": "not_detected"},
                "oracle": {"outcome": "pass"},
                "execution": {
                    "outcome": "signal",
                    "timed_out": False,
                    "health_before": {"healthy": True},
                    "health_after": {"healthy": True},
                },
            },
            {"detector": "not_detected", "oracle": "pass"},
        )
        self.assertFalse(accepted)
        self.assertIn("invalid execution outcome=signal", reasons)

    def test_fault_spec_requires_target_workload_and_exact_mutation(self) -> None:
        workload = validation.WORKLOAD_BY_ID["d128-block"]
        document = {
            "schema_version": 1,
            "target": "gfx1201",
            "workload": workload.id,
            "review_required": False,
            "faults": [
                {
                    "id": "drop",
                    "family": "barrier-drop",
                    "environment": {
                        "RJ_CONSAN_FAULT_DROP_BARRIER": "1",
                        "RJ_CONSAN_FAULT_SITE_IDENTITY": "site-a",
                        "RJ_CONSAN_FAULT_BARRIER_SEQUENCE_IDENTITY": "sequence-a",
                    },
                }
            ],
        }
        with temporary_root() as root:
            path = root / "faults.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            loaded = validation._load_fault(path, "gfx1201", workload, "drop")
        self.assertEqual(loaded["id"], "drop")

    def test_checked_in_gfx1201_fault_reference_covers_the_manifest(self) -> None:
        path = Path(__file__).with_name(
            "consan_validation_faults_gfx1201_reference.json"
        )
        with self.assertRaisesRegex(validation.ValidationError, "reference-only"):
            validation._load_fault(
                path,
                "gfx1201",
                validation.WORKLOAD_BY_ID["qwen-prefill"],
                "barrier-drop",
            )
        for workload in validation.WORKLOADS:
            for fault_id in workload.fault_families:
                fault = validation._load_fault(
                    path,
                    "gfx1201",
                    workload,
                    fault_id,
                    allow_reference=True,
                )
                for profile in validation.PROFILE_IDS:
                    policy, trials = validation._fault_trials(fault, profile)
                    self.assertTrue(trials)
                    self.assertIn(
                        policy.get("detector"),
                        {"detected", "not_detected", "statistical"},
                    )
        qwen = validation.WORKLOAD_BY_ID["qwen-prefill"]
        fault = validation._load_fault(
            path,
            "gfx1201",
            qwen,
            "barrier-drop",
            allow_reference=True,
        )
        policy, trials = validation._fault_trials(fault, "sampled")
        self.assertEqual(policy["minimum_detections"], 1)
        self.assertEqual(
            policy["environment"]["RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE"],
            "256",
        )
        self.assertEqual(len(trials), 32)
        self.assertEqual(
            trials[0], {"RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET": "0"}
        )
        self.assertEqual(
            trials[-1], {"RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET": "31"}
        )


if __name__ == "__main__":
    unittest.main()
