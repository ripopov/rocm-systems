# rocJITsu ConSan

ConSan instruments AMD LDS/shared-memory behavior by intercepting HSA
code-object loads, inspecting final native machine code, and loading a patched
replacement when instrumentation is possible. Current live implementation and
validation target RDNA4/gfx1201; ConSan does not translate between GPU ISAs.

ConSan exposes the SuperCollider flavor and three MOI engines. MOI stands for
**Memory-Ordering Instrumentation**.

- `RJ_CONSAN_FLAVOR=supercollider`: redundant-access/read-back checking with an
  automatic non-trapping mismatch marker;
- `RJ_CONSAN_FLAVOR=moi`, `RJ_CONSAN_MOI_ENGINE=record_replay`: bounded
  records plus host replay;
- `RJ_CONSAN_FLAVOR=moi`, `RJ_CONSAN_MOI_ENGINE=sampled`: bounded statistical
  causal windows; and
- `RJ_CONSAN_FLAVOR=moi`, `RJ_CONSAN_MOI_ENGINE=inline_shadow`: supported-form
  exact GPU shadowing and attributed diagnostics.

The flavor and all three engines select every admitted supported site on
gfx1201 and manage registers and reporting automatically. MOI barriers and
atomics are on by default. Sampled chooses runtime stride 16,384 and offset zero
automatically. Users do not choose a patch count, register, report size,
synchronization switch, or sampling residue for ordinary runs.

## Quick start

```sh
cmake --build "$ROCJITSU_BUILD_DIR" --target rocjitsu_dbi_hooks -j4

export CONSAN_HOOK="$ROCJITSU_BUILD_DIR/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so"

env HSA_TOOLS_LIB="$CONSAN_HOOK" \
  RJ_CONSAN_FLAVOR=moi \
  RJ_CONSAN_MOI_ENGINE=record_replay \
  RJ_CONSAN_LOG=1 \
  ./application
```

MOI Record/Replay is the recommended starting engine. It combines an
inspectable host-side model with low measured end-to-end LLM overhead. Its
retained dynamic history is bounded, so a clean replay is not proof of race
freedom.

For a focused workload known to contain supported sites, add
`RJ_CONSAN_REQUIRE_PATCH=1`. For MOI, `RJ_CONSAN_MOI_REQUIRE_RECORDS=1` asserts
that runtime instrumentation produced visible state. These are acceptance
guards, not tuning controls.

Look for transformed-byte, coverage, and completeness records:

```text
ConSan patch end ... outcome=modified-valid ... patches=N modified=true
ConSan summary ... patches=N modified=true
ConSan coverage ... access=... barrier=... atomic=... fence=...
ConSan analysis verdict ... static_complete=... dynamic_complete=...
```

A passing workload proves its independent oracle was preserved, not that it is
race-free. A workload failure, timeout, signal, or GPU reset is not a ConSan
diagnostic.

## Documents

- [FLAVORS.md](FLAVORS.md): conceptual, phase-by-phase comparison of what the
  flavor and three engines do on the device, defer for later, and do on the
  host.
- [TUTORIAL.md](TUTORIAL.md): current no-tuning commands and safe validation
  workflow.
- [USAGE.md](USAGE.md): public controls, defaults, coverage, and diagnostics.
- [DESIGN.md](DESIGN.md): architecture, implemented behavior, and semantic
  boundaries.
- [SPILLING.md](SPILLING.md): ConSan register selection, ownership, private
  layout, and runtime integration.
- [AMDGPU register spilling](../spilling.md): reusable RocJitsu allocation and
  gfx1201 save/restore backend, including provenance.
- [VALIDATION.md](VALIDATION.md): executable workspace, clean/fault/overhead,
  provenance, and health contract.
- [STATUS_RDNA4.md](STATUS_RDNA4.md): current measured workload × flavor/engine
  results.
- [MALFORMED_INPUT.md](MALFORMED_INPUT.md): finite malformed-input and GPU
  containment contract.
- [FUTURE_WORK.md](FUTURE_WORK.md): post-green release, detector-quality,
  hardening, workload-breadth, and architecture-port DAGs.

The cumulative gfx1201 north-star table is green. This means each admitted cell
has clean correctness, supported-site coverage, exact-fault detection or an
honest qualified miss, bounded termination, health, overhead, memory, and
retained evidence. It does not mean every flavor or engine detects every fault.
The next near-term checkpoint is one-final-tip scripted reproduction; other
GPU architectures remain hardware-deferred.
