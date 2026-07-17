# ConSan tutorial

Start with the **MOI Record/Replay engine**. It is ConSan's recommended engine
for an ordinary investigation: it has an inspectable host-side model, broad
static instrumentation, and the lowest or near-lowest measured end-to-end LLM
overhead on the current RDNA4 workloads. Its retained history is bounded, so a
clean replay is useful evidence but not proof of race freedom.

ConSan patches final native GPU code through the rocJITsu HSA-tools hook. The
application itself does not need to be rebuilt.

## 1. Build and run Record/Replay

Build the hook in an existing out-of-source rocJITsu build, then name it:

```sh
cmake --build "$ROCJITSU_BUILD_DIR" --target rocjitsu_dbi_hooks -j4

export CONSAN_HOOK="$ROCJITSU_BUILD_DIR/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so"
```

Run an application under Record/Replay:

```sh
env HSA_TOOLS_LIB="$CONSAN_HOOK" \
  RJ_CONSAN_FLAVOR=moi \
  RJ_CONSAN_MOI_ENGINE=record_replay \
  RJ_CONSAN_LOG=1 \
  ./application
```

That is the complete ordinary setup. Record/Replay is also the default MOI
engine, but naming it explicitly makes the command and resulting evidence
unambiguous. Do not choose sites, registers, report sizes, barrier/atomic
switches, or other instrumentation resources for a normal run.

If the application needs a non-system ROCm distribution, set its library path
before running it:

```sh
export LD_LIBRARY_PATH="$ROCM_DIST_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## 2. Check the result

At `RJ_CONSAN_LOG=1`, first verify that applicable code was transformed:

```text
ConSan patch end ... outcome=modified-valid ... patches=N modified=true
ConSan coverage ... access=... barrier=... atomic=... fence=...
ConSan analysis verdict ... static_complete=... dynamic_complete=...
```

Record/Replay then reports its bounded host analysis in a line shaped like:

```text
ConSan MOI auto replay ... diagnostics=N conflict=true|false ...
```

`conflict=true` with an attributed replay diagnostic is positive ConSan
evidence. `conflict=false` means only that the retained bounded snapshot did
not expose a conflict. Complete static-site instrumentation does not guarantee
that every dynamic event survived for replay.

## 3. Try another flavor or engine

Use Inline Shadow when immediate supported-form device-side attribution is
worth more device work:

```sh
env HSA_TOOLS_LIB="$CONSAN_HOOK" \
  RJ_CONSAN_FLAVOR=moi \
  RJ_CONSAN_MOI_ENGINE=inline_shadow \
  RJ_CONSAN_LOG=1 \
  ./application
```

Use Sampled for broad statistical campaigns where probabilistic detection is
acceptable:

```sh
env HSA_TOOLS_LIB="$CONSAN_HOOK" \
  RJ_CONSAN_FLAVOR=moi \
  RJ_CONSAN_MOI_ENGINE=sampled \
  RJ_CONSAN_LOG=1 \
  ./application
```

Use SuperCollider for its complementary delayed redundant-observation signal:

```sh
env HSA_TOOLS_LIB="$CONSAN_HOOK" \
  RJ_CONSAN_FLAVOR=supercollider \
  RJ_CONSAN_LOG=1 \
  ./application
```

All ordinary selections automatically instrument every admitted supported
site, allocate registers and reports, and enable supported MOI barrier and
atomic tracking. Sampled also chooses its runtime sampling parameters
automatically.

| Flavor or engine | Useful positive evidence | What a clean run cannot prove |
| --- | --- | --- |
| Record/Replay | Host replay emitted an attributed conflict from its bounded snapshot. | That events outside the retained snapshot were race-free. |
| Inline Shadow | The GPU emitted an attributed supported-form diagnostic. | That unsupported ISA/event forms were race-free. |
| Sampled | A retained statistical campaign emitted sampled conflicts. | That an undetected run or unsampled event was race-free. |
| SuperCollider | The automatic mismatch marker changed. | That the instability was a happens-before race, or that same-value races were absent. |

For the complete conceptual comparison and measured overheads, see
[FLAVORS.md](FLAVORS.md).

## 4. Add assertions for automation

For a focused workload known to contain supported sites and runtime evidence,
the following assertions reject vacuous or incomplete runs:

```sh
export RJ_CONSAN_REQUIRE_PATCH=1
export RJ_CONSAN_MOI_REQUIRE_RECORDS=1
export RJ_CONSAN_MOI_FORBID_OVERFLOW=1
```

For a known-correct clean control, also reject an unexpected diagnostic:

```sh
export RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1
```

These are test assertions, not tuning controls. Apply them only when the
workload's expected outcome is known. Broad applications may legitimately load
helper code objects with no relevant sites.

A workload oracle failure shows that an injected bug manifested, but is not by
itself a ConSan diagnostic. A timeout, signal, or GPU reset is containment
failure, never detection.

## 5. Reproduce the current Qwen result

The validation runner owns exact workload commands and knob hygiene. It assumes
IREE tools and `rocminfo` are in `PATH` and that the workspace layout described
in [VALIDATION.md](VALIDATION.md) exists:

```sh
export CONSAN_VALIDATION_WORKSPACE_DIR=/path/to/validation-workspace
export CONSAN_VALIDATION_TARGET=gfx1201

python3 emulation/rocjitsu/tests/dbi/consan/consan_validation.py doctor

python3 emulation/rocjitsu/tests/dbi/consan/consan_validation.py explain \
  --workload qwen-prefill --profile all --json

python3 emulation/rocjitsu/tests/dbi/consan/consan_validation.py run \
  --workload qwen-prefill --profile all --phase clean \
  --artifact-root /path/to/new-artifacts
```

The `explain` output answers which exact application command, ConSan settings,
automatic flavor/engine defaults, coverage gates, and fault expectations
apply. Its `workload_specific_tuning` array should remain empty for ordinary
clean rows.

Measure no-fault latency against paired baselines with:

```sh
python3 emulation/rocjitsu/tests/dbi/consan/consan_validation.py run \
  --workload qwen-prefill --profile all --phase overhead \
  --include-baseline --artifact-root /path/to/new-overhead-artifacts
```

See [STATUS_RDNA4.md](STATUS_RDNA4.md) for the current workload × flavor/engine
results. The checked-in table is cumulative evidence;
[FUTURE_WORK.md](FUTURE_WORK.md) tracks one-final-tip reproduction.

## 6. Inject a concurrency fault safely

Do not select a fault by a raw global index in retained evidence. Machine-code
identities change when the compiler, workload, or ConSan transformation changes.
The safe workflow is:

1. inventory the exact current workload;
2. review a target-specific JSON spec containing stable site, sequence, and
   destination identities;
3. precommit the expected detector outcome or statistical trial policy;
4. execute the exact-one mutation under process and GPU-health containment; and
5. compare the diagnostic separately from the workload oracle.

The runner exposes this workflow through `inventory` and `fault`:

```sh
python3 emulation/rocjitsu/tests/dbi/consan/consan_validation.py inventory --help
python3 emulation/rocjitsu/tests/dbi/consan/consan_validation.py fault --help
```

The checked-in `consan_validation_faults_gfx1201_reference.json` is historical
reference evidence and is intentionally not executable. Create and review a
fresh spec from current inventory before a live fault run.

Destructive rows must be serialized. Never run more than four GPU jobs in
parallel, and retain pre/post `rocminfo` plus a known-good smoke workload.

## 7. Run focused regressions

Host/synthetic coverage:

```sh
cmake --build "$ROCJITSU_BUILD_DIR" --target rocjitsu_tests -j4

"$ROCJITSU_BUILD_DIR/tests/rocjitsu_tests" \
  --gtest_filter='ConSan*:*SpillManager*'
```

Focused live controls:

```sh
ctest --test-dir "$ROCJITSU_BUILD_DIR" -j4 --output-on-failure \
  -R '^(ConSanSpillHipTest|ConSanInlineShadowTest|ConSanMoiHipTest)\.'
```

The live controls cover clean preservation, forced VGPR spilling, Inline
diagnostics, barrier ordering, atomics, and Sampled publication. Test names and
counts evolve; use CTest discovery rather than copying a historical count.

## Troubleshooting

No ConSan logs:

- verify `HSA_TOOLS_LIB` names the newly built hook;
- verify the process uses an HSA runtime that honors HSA tools; and
- set `RJ_CONSAN_LOG=1` explicitly.

`modified=false` or zero patches:

- inspect `ConSan coverage_site` reasons at a higher log level;
- distinguish semantic unsupported sites from resource or placement failures;
- use the validation runner's manifest/explain view for known workloads.

MOI reports no visible records:

- confirm an automatic report was planned and allocated;
- inspect required/allocated bytes and typed capacity failures;
- remember that automatic Sampled selection can retain no event in a short run;
  and
- use `RJ_CONSAN_MOI_REQUIRE_RECORDS=1` only when non-vacuity is predeclared.

The GPU becomes unhealthy:

- stop launching GPU rows;
- treat the row as containment failure, not detection; and
- recover the device and pass an uninstrumented smoke before continuing.

## Next documents

- [USAGE.md](USAGE.md): complete public controls and result interpretation.
- [DESIGN.md](DESIGN.md): architecture and engine semantics.
- [SPILLING.md](SPILLING.md): ConSan register allocation and private-memory
  integration; [AMDGPU register spilling](../spilling.md) documents the shared
  backend.
- [VALIDATION.md](VALIDATION.md): reproducible workspace and experiment schema.
- [STATUS_RDNA4.md](STATUS_RDNA4.md): current measured results.
- [FUTURE_WORK.md](FUTURE_WORK.md): post-green work DAGs.
