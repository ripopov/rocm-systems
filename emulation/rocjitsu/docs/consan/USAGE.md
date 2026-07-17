# ConSan usage

ConSan instruments final AMD GPU code objects through the rocJITsu HSA-tools
hook. The current live implementation and retained workload evidence target
RDNA4/gfx1201. It does not translate code objects between GPU architectures.

Use [TUTORIAL.md](TUTORIAL.md) for a short walkthrough,
[FLAVORS.md](FLAVORS.md) for a conceptual device/deferred/host comparison,
[VALIDATION.md](VALIDATION.md) for reproducible workload qualification,
[STATUS_RDNA4.md](STATUS_RDNA4.md) for current results, and
[SPILLING.md](SPILLING.md) for ConSan register-resource policy and
[AMDGPU register spilling](../spilling.md) for the reusable backend.

## Build and load the hook

Build in a rocJITsu CMake build directory, never in the source tree:

```sh
cmake --build "$ROCJITSU_BUILD_DIR" --target rocjitsu_dbi_hooks -j4
```

The hook is normally located at:

```sh
export CONSAN_HOOK="$ROCJITSU_BUILD_DIR/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so"
```

Load it into any HSA application with:

```sh
env \
  HSA_TOOLS_LIB="$CONSAN_HOOK" \
  RJ_CONSAN_FLAVOR=moi \
  RJ_CONSAN_MOI_ENGINE=record_replay \
  RJ_CONSAN_LOG=1 \
  ./application
```

`RJ_CONSAN_FLAVOR` has no default. If it is unset, ConSan does not instrument
the application. For most investigations, start with MOI Record/Replay as shown
above.

## Ordinary flavors and engines

ConSan exposes two top-level flavors. MOI contains three engines:

For the execution model behind these short descriptions, see the side-by-side
comparison in [FLAVORS.md](FLAVORS.md).

| Selection | Ordinary `standard-v1` behavior | Primary tradeoff |
| --- | --- | --- |
| `RJ_CONSAN_FLAVOR=moi`, `RJ_CONSAN_MOI_ENGINE=record_replay` | Instrument all admitted supported access, barrier, atomic, and fence sites; allocate an inventory-sized report; replay visible records on the host. | **Recommended starting engine:** clean reference/debug semantics and low measured LLM overhead, but only a bounded dynamic snapshot. |
| `RJ_CONSAN_FLAVOR=moi`, `RJ_CONSAN_MOI_ENGINE=inline_shadow` | Publish exact-shadow cells and bounded diagnostics on the GPU; track admitted barriers and atomics. | Strongest supported-form attribution, with higher overhead. |
| `RJ_CONSAN_FLAVOR=moi`, `RJ_CONSAN_MOI_ENGINE=sampled` | Patch all admitted supported sites; use automatic runtime stride 16,384 and offset zero; retain bounded sampled causal windows and synchronization metadata. | Low measured LLM overhead, probabilistic detection. |
| `RJ_CONSAN_FLAVOR=supercollider` | Duplicate/read-back supported LDS accesses, delay, compare, and set an automatically allocated non-trapping mismatch marker. | Complementary value-instability diagnostic; it does not attribute a happens-before edge or exact racing pair. |

`RJ_CONSAN_MOI_ENGINE` defaults to `record_replay` when the flavor is `moi`,
but spelling the engine explicitly makes commands and artifacts auditable.

Record/Replay's complete static-site instrumentation is not an exhaustive
dynamic trace: repeated executions of a static site can overwrite earlier
evidence. A clean replay therefore remains inconclusive.

Ordinary runs do not need a register number, report-buffer size, barrier
switch, atomic switch, or sampling setting. The hook logs
`moi_profile=standard-v1` and whether a value came from the standard settings or
an expert override.

## Minimal commands

Recommended default, Record/Replay:

```sh
env HSA_TOOLS_LIB="$CONSAN_HOOK" \
  RJ_CONSAN_FLAVOR=moi \
  RJ_CONSAN_MOI_ENGINE=record_replay \
  RJ_CONSAN_LOG=1 \
  ./application
```

Inline Shadow:

```sh
env HSA_TOOLS_LIB="$CONSAN_HOOK" \
  RJ_CONSAN_FLAVOR=moi \
  RJ_CONSAN_MOI_ENGINE=inline_shadow \
  RJ_CONSAN_LOG=1 \
  ./application
```

Sampled:

```sh
env HSA_TOOLS_LIB="$CONSAN_HOOK" \
  RJ_CONSAN_FLAVOR=moi \
  RJ_CONSAN_MOI_ENGINE=sampled \
  RJ_CONSAN_LOG=1 \
  ./application
```

SuperCollider:

```sh
env HSA_TOOLS_LIB="$CONSAN_HOOK" \
  RJ_CONSAN_FLAVOR=supercollider \
  RJ_CONSAN_LOG=1 \
  ./application
```

Add acceptance guards only when their expected outcome is known:

```sh
export RJ_CONSAN_REQUIRE_PATCH=1
export RJ_CONSAN_MOI_REQUIRE_RECORDS=1
export RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1   # known-correct MOI control
export RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS=1  # predeclared positive MOI control
export RJ_CONSAN_MOI_FORBID_OVERFLOW=1
```

Do not enable both diagnostic guards. `REQUIRE_PATCH` is useful for focused
workloads but can be too strict for a broad application that loads helper code
objects with no admitted sites.

## Core controls

| Variable | Default | Meaning |
| --- | --- | --- |
| `RJ_CONSAN_FLAVOR=supercollider|moi` | unset | Select instrumentation. |
| `RJ_CONSAN_MOI_ENGINE=record_replay|sampled|inline_shadow` | `record_replay` for MOI | Select the MOI engine. |
| `RJ_CONSAN_LOG=N` | disabled | Enable compact logs at `1`; larger values add inventory detail. |
| `RJ_CONSAN_FAIL_CLOSED=0|1` | `0` | Reject unsupported/invalid transformation outcomes instead of loading the original. |
| `RJ_CONSAN_REQUIRE_PATCH=0|1` | `0` | Reject an applicable code object when no real access/barrier/atomic/fence instrumentation patch is emitted. Prologues and metadata-only changes do not satisfy it. |
| `RJ_CONSAN_FLAT_PROVENANCE=likely|strict` | `likely` | Admit proven `Group` plus heuristic `MaybeGroup` flat LDS sites, or only proven `Group` sites. |
| `RJ_CONSAN_DUMP_DIR=PATH` | unset | Write original and transformed `.hsaco` objects for inspection. |

## SuperCollider controls

| Variable | Default | Meaning |
| --- | --- | --- |
| `RJ_CONSAN_SC_REPORT_MODE=auto|trap` | `auto` | `auto` owns a non-trapping sticky marker per relevant code object. `trap` is an expert process-disrupting mode. |
| `RJ_CONSAN_REPORT_BUFFER=0xADDR` | unset | Use a caller-owned device-visible 32-bit marker instead of automatic allocation. |
| `RJ_CONSAN_REPORT_MARKER=N` | `1` | Value written on mismatch. |
| `RJ_CONSAN_DELAY=N` | `0` | Delay parameter between the guest access and duplicate/read-back. |
| `RJ_CONSAN_DELAY_MODE=nop|sleep|sleep_var` | `nop` | Select `s_nop`, `s_sleep`, or `s_sleep_var` delay lowering. |
| `RJ_CONSAN_DELAY_VAR_SSRC=N` | `106` | Scalar source encoding used by `sleep_var`. |
| `RJ_CONSAN_CHECK_TRAP_MODE=all|lds|flat` | `all` | Restrict SuperCollider to native DS or admitted flat LDS sites for debugging. |

The automatic marker reports that at least one duplicated/read-back value
differed. It does not identify an address, lane, value, or happens-before
violation. A race-free program can advance another wave between the original
and repeated access, so interpret the marker only through a bounded
clean/fault differential.

## MOI report buffers

With no caller-owned buffer, the hook first inventories the final transformed
code and computes the exact report layout required by the selected ordinary
engine:

- Record/Replay reserves admitted static access ranges and enabled barrier,
  atomic, fence, and diagnostic regions.
- Sampled reserves admitted logical ranges, bounded window banks,
  synchronization metadata, and pending-acquire state.
- Inline Shadow reserves one versioned exact-shadow slot per four-byte cell in
  the maximum declared LDS span of the owners plus the enabled diagnostic and
  ordering regions.

The automatic allocator requests the exact planned bytes. It never silently
shrinks site coverage or disables an event kind to fit. The hard ceilings are
16 MiB per automatic buffer and 256 MiB live automatic-report memory per
process. Arithmetic overflow, a ceiling violation, or allocation failure is a
typed incomplete outcome.

| Variable | Default | Meaning |
| --- | --- | --- |
| `RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE=N` | 16 MiB ceiling | Expert cap for HSA-tool-owned allocation; ordinary inventory still requests exact bytes below the cap. `0` disables automatic allocation. Dynamic access append requires an explicit finite cap. |
| `RJ_CONSAN_MOI_REPORT_BUFFER=0xADDR` | unset | Caller-owned device-visible report buffer. |
| `RJ_CONSAN_MOI_REPORT_BUFFER_SIZE=N` | `0` | Size of the caller-owned buffer; layout requirements depend on the engine and enabled event families. |
| `RJ_CONSAN_MOI_REQUIRE_RECORDS=0|1` | `0` | At unload, require some visible auto-buffer access, synchronization, shadow, or sampled evidence. |
| `RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS=0|1` | `0` | Require at least one Inline, replay, or sampled diagnostic/conflict. |
| `RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=0|1` | `0` | Require zero diagnostics/conflicts. |
| `RJ_CONSAN_MOI_REQUIRE_REPLAY_CONFLICT=0|1` | `0` | Record/Replay-only positive guard. |
| `RJ_CONSAN_MOI_FORBID_OVERFLOW=0|1` | `0` | Fail if evidence was truly dropped. Sampled bounded saturation is reported separately from loss. |

The unload summary and validation JSON retain required/allocated bytes,
per-region capacities, current and peak live bytes, private spill growth,
Inline LDS shadow size, Sampled banks, saturation, undercoverage, overflow,
and drops.

## MOI event and sampling controls

| Variable | Default | Meaning |
| --- | --- | --- |
| `RJ_CONSAN_MOI_TRACK_BARRIERS=0|1` | `1` for every MOI engine | Track admitted barrier events. Explicit `0` is an expert compatibility override. |
| `RJ_CONSAN_MOI_TRACK_ATOMICS=0|1` | `1` for every MOI engine | Track admitted atomic/fence ordering evidence. Explicit `0` is an expert compatibility override. |
| `RJ_CONSAN_MOI_DYNAMIC_ACCESS_RECORDS=0|1` | `0` | Record/Replay per-lane dynamic append. This is bounded expert tracing, not an exhaustive ordinary contract. |
| `RJ_CONSAN_MOI_SAMPLE_STRIDE=N` | `1` | Sampled static site stride; this removes nonselected sites and therefore limits declared coverage. |
| `RJ_CONSAN_MOI_SAMPLE_OFFSET=M` | `0` | Static residue, smaller than the static stride. |
| `RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE=N` | `16,384` for Sampled | Expert power-of-two runtime stride in `1..16777216`; leaves all eligible static sites patched. |
| `RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET=M` | `0` | Expert runtime residue smaller than the runtime stride. |
| `RJ_CONSAN_MOI_SAMPLED_CHECK=0|1` | `0` | Enable the lower-fidelity immediate adjacent-range GPU check in addition to host scanning. |

Sampled runtime selection mixes hardware dispatch identity, workgroup
coordinates, wave owner, epoch, persistent per-wave sequence, static site, and
LDS address. Each logical range receives as many as eight immutable windows
when capacity permits. A later valid identity after every bank fills is
`sampled_saturated_windows`; malformed publication or true evidence loss uses
separate counters and makes strict acceptance incomplete.

## Resource overrides

Register selection is automatic. Access, barrier, atomic, and diagnostic
probes use dead VGPRs, fresh descriptor-backed windows, or spill-preserved
windows as appropriate. Scalar windows are placed above decoded and metadata
ownership and preserve EXEC, VCC, and SCC. These variables are expert/debug
overrides, not ordinary setup:

- `RJ_CONSAN_TMP_VGPR`
- `RJ_CONSAN_MOI_EXEC_SAVE_SGPR`
- `RJ_CONSAN_MOI_OWNER_VGPR`
- `RJ_CONSAN_MOI_EPOCH_VGPR`
- `RJ_CONSAN_MOI_OWNER_SGPR`
- `RJ_CONSAN_MOI_OWNER_SOURCE=workitem_id|hw_id` (defaults to `workitem_id`)
- `RJ_CONSAN_MOI_INIT_OWNER_EPOCH`

Overrides remain subject to ownership, alignment, liveness, overlap, and
descriptor checks. They cannot force an unsafe plan. See
[SPILLING.md](SPILLING.md).

## Malformed-input guard

`RJ_CONSAN_ABORT_UNMATCHED_BARRIER_WAIT=1` is an opt-in destructive containment
guard. It replaces only a statically unique immediate wait which belongs to no
bounded same-owner barrier sequence with a terminal instruction and records an
`inline-malformed-barrier-abort` patch. Dynamic, ambiguous, and paired waits
are untouched. It is not enabled by ordinary flavors or engines and is not a
race diagnostic. See [MALFORMED_INPUT.md](MALFORMED_INPUT.md).

## Fault injection

Fault controls mutate final native code before instrumentation. Retained
evidence must use the validation workflow rather than manual index selectors:

1. inventory the exact committed binary;
2. review stable site/sequence/destination identities;
3. require exactly one mutation;
4. run the clean/fault pair under process and GPU-health containment; and
5. classify a ConSan diagnostic separately from workload corruption, timeout,
   signal, or device loss.

The supported injector families include barrier drop/move/ID-scope/participant
changes, atomic wrong address/order/scope, and ordinary access
address/order/scope. `RJ_CONSAN_FAULT_*`, `RJ_CONSAN_SC_PERTURB_*`,
`RJ_CONSAN_PROBE_*`, `RJ_CONSAN_TEST_*`, and
`RJ_CONSAN_MOI_PARTITION_MASK_DEBUG` are experiment implementation controls,
not a stable hand-authored user interface. Use
`consan_validation.py explain --json` to audit every effective setting and
`inventory`/`fault` with a reviewed spec to execute it. Historical reference
specs are deliberately non-executable.

## Coverage and diagnostics

At `RJ_CONSAN_LOG=1`, the important records are:

```text
ConSan patch end ... outcome=... patches=... modified=...
ConSan summary ... patches=... modified=...
ConSan coverage ... flavor=... engine=... access=... barrier=... atomic=... fence=...
ConSan coverage_site ... kind=... disposition=... outcome=... reason=... lowering_reason=... resource_reason=...
ConSan analysis verdict ... static_complete=... dynamic_complete=...
ConSan MOI report memory required_bytes=... allocated_bytes=... peak_live_bytes=...
```

For each event kind, the aggregate accounting is:

```text
discovered = supported + unsupported
supported = selected + expert_limit_omitted
selected = patched + resource_failed + placement_or_lowering_failed
```

An unsupported-only object remains applicable and incomplete. MOI emits one
typed `coverage_site` row per relevant final-code site; the independent Python
gate reconciles row counts and stable vocabularies rather than trusting producer
booleans. SuperCollider exposes strict aggregate coverage but does not
manufacture MOI per-site dispositions.

Interpret outcomes independently:

- `modified=true` proves replacement bytes were loaded, not that a race was
  found.
- A passing workload proves non-corruption of that oracle, not race freedom.
- A workload output mismatch proves a fault manifestation, not a ConSan
  diagnostic.
- A timeout, signal, GPU reset, or failed health check is containment failure,
  never detection.
- A Sampled clean run or statistical miss is inconclusive about race freedom.

## Reproducible validation

Set a workspace containing the external repositories/builds described in
[VALIDATION.md](VALIDATION.md), place IREE tools and `rocminfo` in `PATH`, and
run:

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

Use the `overhead` phase with `--include-baseline` for correct-workload latency.
Use `inventory` followed by `fault --spec REVIEWED.json` for mutation evidence.
Never execute the checked-in historical reference spec as a live plan.

GPU tests must use at most four parallel jobs. Serialize destructive fault
rows and retain pre/post device health.

Focused local regressions:

```sh
cmake --build "$ROCJITSU_BUILD_DIR" --target rocjitsu_tests -j4

"$ROCJITSU_BUILD_DIR/tests/rocjitsu_tests" \
  --gtest_filter='ConSan*:*SpillManager*'

ctest --test-dir "$ROCJITSU_BUILD_DIR" -j4 --output-on-failure \
  -R '^(ConSanSpillHipTest|ConSanInlineShadowTest|ConSanMoiHipTest)\.'
```

## Current boundaries

- Live native validation is gfx1201-only. Other target ISAs need their own
  encoders, inventories, exact mutation identities, and GPU evidence.
- ConSan is LDS/shared-memory focused. Selected atomics/fences provide ordering
  evidence; they are not general global-memory race instrumentation.
- Flat/generic LDS classification is conservative. `MaybeGroup` is heuristic;
  use `strict` when precision matters more than recall.
- SuperCollider reports redundant-access instability, not causality.
- Record/Replay is a bounded snapshot unless dynamic append is explicitly
  enabled, and dynamic append is still bounded by its finite report.
- Sampled is probabilistic and can miss races.
- Inline Shadow has bounded diagnostics and supported-form semantics, not
  unbounded tracing or complete ISA coverage.
- Ordinary VGPR spilling exists on gfx1201; general SGPR and AccVGPR spilling
  do not. Unsupported ownership or resource shapes fail explicitly.

See [STATUS_RDNA4.md](STATUS_RDNA4.md) for measured workload coverage,
diagnostics or qualified misses, overhead, and memory evidence.
