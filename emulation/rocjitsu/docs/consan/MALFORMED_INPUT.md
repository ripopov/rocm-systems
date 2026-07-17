# ConSan malformed-input containment contract

This document defines ConSan's finite malformed-input and containment contract.
It is deliberately separate from detection quality: a timeout, crash, reset,
or workload failure is never a ConSan race diagnostic. Extensions to this
contract are tracked by the hardening DAG in
[FUTURE_WORK.md](FUTURE_WORK.md).

The contract applies to SuperCollider and to all three MOI engines:
`record_replay`, `inline_shadow`, and `sampled`. It covers the public transform
boundary, the HSA hook's install decision, and carefully contained execution of
structurally valid but semantically ill-formed GPU programs. Arbitrary
structurally malformed bytes must never be submitted to a GPU.

## Required outcomes

Every transform invocation must terminate within its budget and produce
exactly one typed, transactional outcome:

| Transform outcome | Replacement bytes | Fail-open install | Fail-closed install |
| --- | --- | --- | --- |
| `ModifiedValid` | Nonempty, independently validated, and accompanied by a nonempty patch ledger | replacement | replacement |
| `Unchanged` | Empty | original | original |
| `Unsupported` | Empty | original | reject |
| `Invalid` | Empty | original | reject |

A non-`ModifiedValid` result with replacement bytes or patches is a contract
violation. A `ModifiedValid` result that fails final structural validation is a
contract violation and must be rejected under both policies. The loader must
make its decision from the typed result and final-validation bit; it must not
infer success from a nonempty byte vector.

`Unchanged` is not proof that an input is well formed. For example, a malformed
symbol record which is irrelevant to the selected transform may remain outside
the transformer's semantic validation boundary. Such a row is retained as an
explicit original-load outcome, not relabeled as rejection. Expanding the
semantic validation boundary is separate work; the contract must accurately
report what is and is not rejected today.

## Frozen host corpus

The tight-loop host matrix uses one deterministic, supported gfx1201 ELF seed
containing a native LDS store and runs every case through all four engine
profiles.

### Structural fixtures

The named fixture set is:

1. a truncated eight-byte instruction;
2. a truncated ELF header;
3. a truncated section table;
4. a zero section count;
5. a section table outside the image;
6. a text range outside the image;
7. a zero symbol-table entry size;
8. a kernel symbol with an invalid section index;
9. an overflowing kernel-symbol range;
10. a truncated kernel descriptor;
11. a misaligned kernel-descriptor file offset;
12. a descriptor entry point outside text;
13. partially overlapping function symbols;
14. an allocated section whose alignment would require excessive text-growth
    padding; and
15. a program-header table whose file range overflows.

The zero-section-count fixture is the permanent regression for the prior
zero-length `memcpy` sanitizer finding. Descriptor byte access at a deliberately
misaligned file offset is the permanent regression for the prior unaligned
typed-access finding. The excessive-alignment fixture is the regression for a
compact valid-ELF fuzz mutation that previously reached `replace_text` and
requested an astronomically large padding allocation. The
overflowing-program-header fixture covers the subsequent compact-seed finding
where a debug-only table-bounds assertion became an out-of-bounds `memcpy` in
a release sanitizer build.

### Bounded mutation property

For the same seed, the matrix transforms:

- every prefix length from zero through the full image; and
- one deterministic nonzero XOR mutation at every byte offset.

Each result is checked against the complete transaction and install table
above. This is a bounded property test, not a claim that all ELF mutations have
been enumerated.

### Runtime-state fixtures

Host unit tests retain malformed or unstable report-buffer states separately
from input-ELF corruption. The required families are incomplete/changed
generation snapshots, reserved bits and invalid versions, malformed owner or
epoch identity, capacity drift, publication collisions, malformed sampled
windows, malformed inline release/token claims, and dropped or unsupported
records. These must fail closed for analysis: they may never be interpreted as
proof of a clean execution.

## Retained sanitizer/fuzzer corpus

Four profile-specific transform fuzzers call the public transform boundary
under SuperCollider, Record/Replay, Inline Shadow, and Sampled respectively and
assert the same transaction table. One libFuzzer input is exactly one public
transform invocation, so the 10-second watchdog measures the API operation it
is intended to contain rather than an artificial serial batch of four. Each
profile receives the full 10,000-event budget: 40,000 public transforms total,
the same transform count as 10,000 callbacks of the former aggregate harness.
The transform evidence is deliberately stratified:

- a fixed replay tier executes the original large production object, each
  previously transformed production object, and every retained crash/timeout
  regression once through every profile under the 10-second per-input limit;
  and
- a mutation tier starts from the minimized malformed regression plus a
  compact valid gfx1201 object with a real padded LDS access, then runs all
  10,000 libFuzzer events per profile.

The tiers prevent libFuzzer from filling its evolving corpus with hundreds of
roughly 0.5 MiB decoder-feature variants and spending the event budget on
near-duplicate production images. They do not omit the large objects: every
one remains an exact sanitizer replay input for every profile, while the
compact valid seed keeps parser, decoder, CFG, placement, replacement, and
final-validation paths reachable at sustained mutation throughput.
The placement fuzzer covers bounded overlapping inline, local-cave, and
appended-cave requests. A qualifying campaign uses a Clang ASan+UBSan build and
retains:

- the exact source revision and CMake cache;
- original and previously transformed gfx1201 code-object seeds;
- every minimized crash or timeout input;
- a sorted SHA-256 manifest of the starting and ending corpus;
- the exact command line and random seed;
- final coverage/features, event count, elapsed time, and peak RSS; and
- empty-or-hashed artifact directories proving whether a failure was emitted.

The finite acceptance budgets are:

| Target | Event budget | Maximum input | Per-input timeout | Campaign wall budget | Peak RSS budget |
| --- | ---: | ---: | ---: | ---: | ---: |
| transform mutation tier (each of four profiles) | 10,000/profile | 1 MiB | 10 s | 20 min aggregate | 1.5 GiB/process |
| transform fixed replay tier | every retained input/profile pair once | 1 MiB | 10 s | included above | 1.5 GiB/process |
| placement | 100,000 | 4 KiB | 10 s | 2 min | 1.0 GiB |

Any ASan/UBSan report, invariant abort, libFuzzer timeout, unexpected signal,
or newly emitted crash artifact fails the campaign. All four profile campaigns
must pass; a missing profile fails the aggregate. Reaching the event budget
cleanly proves only this exact corpus and event budget.

## Time and complexity budgets

The focused structural and bounded-mutation matrix must complete as one host
process within 10 seconds in the ordinary RelWithDebInfo build. CI or a local
acceptance runner must enforce that deadline outside the process; a test that
eventually returns after the deadline is a failure. Each fuzz input is capped
at 1 MiB and receives libFuzzer's 10-second timeout. No acceptance command may
silently remove either bound.

This is presently an external deadline, not an internal transform watchdog.
Therefore a clean unit run proves the frozen cases terminate, while the fuzzer
campaign broadens that finite evidence. It does not prove termination for all
possible byte strings.

## Hook and loader matrix

Qualification requires more than direct patcher tests. For every engine,
retained hook tests must cover:

1. invalid/unsupported input under fail-open, loading the untouched original;
2. the same typed input under fail-closed, rejecting before runtime load;
3. a valid modified replacement, loading only the independently validated
   final image; and
4. a corrupted or failed-validation replacement, rejecting rather than falling
   back to a partially modified image.

Direct unit tests of `consan_install_action` are necessary evidence for this
table, but actual hook-facing replacement-reader and loader outcomes remain
part of the qualification contract.

## Valid-ELF ill-formed GPU controls

Only a separate serialized gfx1201 campaign may execute structurally valid
programs with divergent barriers, unmatched split-barrier signal/wait, or
other intentionally ill-formed synchronization. Each row requires:

- an isolated process group and a strict three-second row timeout;
- at most one GPU row in flight (and never more than four GPU jobs globally);
- pre-row and post-row `rocminfo` device visibility;
- a post-row known-good gfx1201 smoke workload;
- automatic quarantine after device loss or failed health smoke; and
- separate outcome fields for workload result, timeout/signal, ConSan
  diagnostic, and device health.

Killing a host process does not establish that an already submitted kernel was
aborted. A row that reaches its timeout is therefore a containment failure for
this contract, even when the device subsequently appears healthy. The existing
runner mechanism is retained as a safety boundary, but full acceptance requires
every frozen GPU control to terminate without timeout or reset and leave the
device healthy.

## Current boundary

The host regression suite distinguishes deadline, signal, ordinary failure,
preflight health loss, and postflight device loss. A timed-out or unhealthy GPU
row creates a quarantine which an unhealthy recovery probe cannot clear. The
global GPU lock covers both the authoritative quarantine recheck and quarantine
clearing, so a queued row cannot launch after a preceding row loses the device.

Every hook flavor supports the same narrow opt-in unmatched-wait guard through
`RJ_CONSAN_ABORT_UNMATCHED_BARRIER_WAIT=1`. Only a unique immediate wait absent
from every bounded multi-event barrier sequence is rewritten to `s_endpgm`;
its final image carries the `inline-malformed-barrier-abort` patch and a
runner-visible static diagnostic. Bounded non-adjacent pairs are associated
before this decision. Dynamic or ambiguous waits fail open rather than being
guessed malformed, and deliberate barrier fault injection disables the guard
during mutation and instrumentation.

Transactional host rejection does not prove GPU containment, and post-timeout
device health does not turn a hang into a successful diagnosis. A qualifying
release therefore needs both the bounded host/sanitizer corpus and the
serialized valid-ELF GPU controls. Further final-image reconciliation, corrupt
runtime-state coverage, fuzzing, and multi-wave/multi-launch qualification are
the `H` track in [FUTURE_WORK.md](FUTURE_WORK.md).
