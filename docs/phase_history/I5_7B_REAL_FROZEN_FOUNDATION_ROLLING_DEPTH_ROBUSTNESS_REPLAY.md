# I5.7B - Real Frozen-Foundation RollingDepth Robustness Replay

## Scope

I5.7B is the real production-native execution step authorized by the closed I5.7A admission artifact. It executes the existing `RollingDepthProductionProviderAdapter` sequentially on exactly six admitted Castle scenarios and records immutable provider-native, alignment, fusion, timing, resource and integrity evidence.

I5.7B is not a provider integration, redesign, ranking, Auto policy, Hybrid experiment, MP-SfM experiment, metric-calibration step, or OOM/crash-recovery redesign.

## Frozen parent

The runner requires the exact closed I5.7A formal admission SHA256:

`040ee6ce45f32e9f3f51087ce5f5b8764ced9a33e978d8024d04867efa5fae61`

Through that artifact it reuses and verifies the exact I5.6A authoritative PyCOLMAP SfM report bindings. No SfM or bundle adjustment is rerun and source material is not rematerialized.

## Real replay matrix

The execution order is frozen:

1. `castle_control` - temporal sequence control, 19 registered views.
2. `castle_low_overlap` - temporal support/spacing stress, 6 registered views; not an MVS-overlap equivalence claim.
3. `castle_weak_parallax` - narrow-baseline temporal geometry stress, 7 registered views.
4. `castle_overexposure` - photometric overexposure temporal stress, 19 registered views.
5. `castle_symmetry` - registered authoritative subset only, 12 views.
6. `castle_dynamic_proxy` - dynamic/motion-boundary temporal stress, 19 registered views.

Every scenario uses the complete registered sequence in source temporal order. I5.7B deliberately bypasses adaptive view reduction for this admitted scientific replay and makes an explicit, no-fallback selection of `rolling-depth-temporal`.

## Runtime and production defaults

Before inference the runner verifies the existing RollingDepth external runtime, runtime lock, exact source revision, checkpoint, CUDA availability and descriptor. It preserves `RollingDepthProductionConfig()` unchanged, including the current `cuda:0`, `fp16`, 512 processing resolution, `[1,25]` dilations, max VAE batch size 1, snippet unloading, and no permanent global view cap.

Preflight mode performs all frozen-foundation and runtime checks without inference. Production-native mode must use a separate new evidence directory.

## Scientific terminal outcomes

`pass`, `degraded`, and `rejected` are valid terminal scientific outcomes. `rejected` is admitted only when the existing one-global-sequence alignment quality gate returns `global_sequence_alignment_rejected:*`. A rejection is preserved and is never automatically retried or converted to an infrastructure failure.

Unexpected runtime, worker, file-integrity, malformed-evidence or other exceptions are not scientific provider verdicts. The runner writes `i5_7b_attempt_failure.json`, preserves the attempt, exits non-zero, and performs no retry-until-pass.

## Narrow recovery after the first real wrapper-schema failure

The first real I5.7B `castle_control` execution completed RollingDepth provider work and produced native/canonical artifacts, but the initial I5.7B evidence consumer incorrectly looked for authority fields directly under `execution_result` instead of under the real `DenseProviderExecutionResult.as_dict()` structure at `execution_result.authority`. The preserved failure was therefore a wrapper/evidence-schema failure, not a RollingDepth scientific rejection.

The correction reads the real nested authority contract and provides a deliberately narrow `recovery-production-native` mode. Recovery requires the preserved original attempt directory, a SHA256 manifest whose own SHA256 is supplied explicitly, and the preserved console log with its exact SHA256. Every manifest-bound file is rehashed; paths outside the preserved attempt are rejected; both failure artifacts must carry the exact known non-scientific wrapper exception; and the frozen `castle_control` SfM/source identity is revalidated.

Recovery never copies, edits or reruns the original provider output. It creates a new timestamped recovery attempt, writes a recovered `castle_control` scenario record that references the original immutable artifacts, records `provider_execution_rerun=false`, and then executes only scenarios 2-6. Because the original wrapper failed before persisting its in-memory wall-time/VRAM/RAM observations, those three fields remain explicitly `null` and all control-relative performance ratios that depend on them remain missing. They must never be reconstructed from a rerun.

The correction also binds the native evidence at the actual production layout `provider/provider/...`, including `worker_output/rolling_depth_evidence.npz`.

## Evidence and resources

Each real scenario preserves:

- frozen SfM report identity and SHA256;
- source/SfM integrity before and after provider execution;
- provider execution summary;
- native worker manifest/depth evidence when produced;
- one-global-sequence affine alignment evidence;
- canonical fused NPZ/PLY when alignment/fusion is accepted;
- wall time;
- best-effort device-wide GPU-0 VRAM and system-wide host-RAM observations.

Resource observations are descriptive evidence, not qualification thresholds. Missing observations stay `null`; they are never fabricated.

## Authority

PyCOLMAP remains the sole camera/scene-frame authority. RollingDepth does not gain camera-pose authority, physical metric scale, measurement authority, certified metrology, mesh authority or provider-promotion status. The global affine maps relative RollingDepth depth only into the arbitrary authoritative PyCOLMAP reconstruction gauge.

## Explicit non-execution

I5.7B does not run PyCOLMAP SfM, COLMAP MVS, DA3, Video Depth Anything, MP-SfM, Hybrid or Auto. It does not change production thresholds/defaults and does not compare providers for a winner.

## Next boundary

A complete six-scenario I5.7B evidence artifact authorizes only I5.7C - RollingDepth robustness qualification and closure. I5.7C is evidence-only and must not rerun the provider merely to improve a result.
