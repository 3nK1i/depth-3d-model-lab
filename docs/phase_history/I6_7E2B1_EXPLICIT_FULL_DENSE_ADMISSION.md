# I6.7E2B1 — Explicit final dense-result admission boundary

## Baseline and intent

The I6.7D source checkpoint is frozen and independently SHA256-verified. I6.7E1 inspected all 1,686,160 RGB vertices in the separate, frozen real I6.7B `fused.ply` and verified its digest; that standalone artifact **does not** belong to an admitted One Start production run and must never be silently attached to an unrelated session. I6.7E2A introduced a pure application-layer typed candidate bound to an *already admitted* sparse One Start result, the exact PyCOLMAP SfM frame, one explicitly selected completed COLMAP-MVS provider envelope and the independently hashed full artifact.

## Implementation

`app/reconstruction_dense_final_admission.py` adds an explicit, single-commit admission boundary. Its caller must supply the E2A candidate **and the original complete E2A inputs**, not merely a preview, a report string or a potentially forged candidate. At admission time it reruns the E2A checks and full-vertex PLY scan, checks candidate equality and artifact identity, and writes exactly one no-overwrite admission manifest at:

`<admitted-sparse-output>/dense_runs/<dense-run-uuid>/final_dense_admission.json`

The manifest binds sparse session/result SHA, frame, concrete selected provider, run UUID, full geometry path/hash/bytes/vertex count and bounds. A temporary file is written and fsynced; a same-directory hard link publishes the final entry atomically *without overwriting an existing admission*. Failures preserve previous accepted results; a post-commit error is **not** a reason to rerun until success. Reading requires the originating candidate, exact manifest bytes/identity, and a new bounded full-vertex PLY revalidation; do not load geometry into the application from a stale in-memory 'admitted' flag.

The result admits a completed dense *artifact for subsequent qualified consumption*, **not** a persistent Digital Twin scene. It grants neither new camera-pose/physical scale/measurement authority nor UI/GPU readiness, and does not change the frozen 4,096-point provisional preview. Work must run off the Qt GUI/render thread because full PLY verification may be substantial. Core remains independent of `app`/Qt.

## Deliberate remaining E2B work

**This increment is the final-artifact admission component, not live product wiring.** Neither One Start nor the UI calls it yet. The subsequent E2B increment must connect the admitted sparse result to a truly isolated, explicitly selected production dense worker with session-bound cancellation, bounded progressive transport, independent completed-provider output SHA and a call to this admission API. Only after that binding and full-result viewer qualification may the app display all dense points as a final result. An I6.7B qualification artifact must not be misrepresented as a new One Start run. Production Auto, mesh, metric scale, metrology, and provider policy remain unchanged.

## Validation

The focused gate runs 12 new unit/integration tests plus the core/UI architectural guard. It tests completed full-artifact admission and read-back, no-overwrite idempotence rejection, foreign candidate/source, degraded provider, artifact corruption before/after acceptance, symlinked admission destination, manifest mutation, oversized receipt, concurrent no-overwrite admission, and a fake isolated MVS worker whose provisional snapshot sink fails while its independently completed artifact still qualifies. The closure gate also runs E2A/E1 and frozen A–D predecessor regressions, storage preflight and the full unit suite. No actual PatchMatch/fusion, model inference, UI changes or Gitea/GitHub operations are part of this increment.
