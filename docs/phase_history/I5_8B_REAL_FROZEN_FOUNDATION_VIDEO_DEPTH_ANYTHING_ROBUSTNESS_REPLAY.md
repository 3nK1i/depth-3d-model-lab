# I5.8B — Real Frozen-Foundation Video Depth Anything Robustness Replay

## Status

Implementation gate complete. Real production evidence is intentionally generated only in the authoritative `depth` environment after the overlay is applied and independently validated.

## Purpose

I5.8B executes the six scenarios admitted by I5.8A through the existing I4.3D1 `VideoDepthAnythingProductionProviderAdapter` against the already-frozen PyCOLMAP SfM foundations. It does not redesign or retune Video Depth Anything (VDA). The goal is to preserve the native production behavior and record provider-specific temporal robustness evidence for later I5.8C qualification.

## Frozen predecessor

I5.8B accepts only the closed I5.8A admission artifact with SHA256:

`1d635cdcefe02e1ee3b66dfaa8af4df49fb952e58cbf4446b50c2d562369d90a`

The nested I5.6A Castle SfM admission/report bindings are re-hashed and re-analyzed before any provider execution. Each scenario's authoritative source/SfM identity is revalidated again after provider execution.

## VDA-specific production semantics

I5.8B deliberately does not reuse RollingDepth's sequence-global alignment semantics. The replay freezes the actual VDA production contract:

- explicit provider ID `video-depth-anything-temporal`;
- official stateful streaming-cache sequence behavior;
- complete registered sequence in source temporal order;
- stable pixel grid required by the I4.3D1 production adapter;
- warm-up state reset before the user sequence by the native worker path;
- one worker frame record per registered view;
- native frame latency evidence retained descriptively;
- independent per-registered-frame affine alignment into the authoritative PyCOLMAP SfM gauge;
- no sequence-global affine alignment;
- model-native metric depth retained as model evidence only;
- model-estimated intrinsics retained as model evidence only;
- PyCOLMAP remains the camera-pose/global reconstruction-frame authority.

## Hardware-resolved runtime

Before any production replay, I5.8B performs an execution-free preflight that:

1. detects the current GPU through the production hardware detector;
2. resolves `ComputeProfile.AUTO` through the existing profile resolver;
3. resolves the VDA runtime config through `MODEL_REGISTRY`;
4. converts the production VDA source/checkpoint identities to exact absolute paths;
5. probes the actual VDA environment;
6. requires CUDA readiness for production-native replay;
7. records the resolved profile, encoder, device, precision, source revision and checkpoint SHA256;
8. verifies that the checkpoint is the checkpoint selected by the resolved production profile.

I5.8B does not force the RTX 2060/4090/5090 to a common model size. The production hardware resolver remains authoritative. The exact resolved runtime is frozen into the evidence and must remain identical across all six scenarios in one campaign.

## Admitted scenarios

I5.8B uses the six I5.8A-admitted frozen Castle source/SfM bindings:

| Scenario | Registered / Input | VDA interpretation |
|---|---:|---|
| `castle_control` | 19 / 19 | streaming-cache registered-sequence control |
| `castle_low_overlap` | 6 / 7 | registered-keyframe spacing/cache-continuity stress; not MVS-overlap equivalence |
| `castle_weak_parallax` | 7 / 7 | narrow-baseline streaming temporal-stability stress |
| `castle_overexposure` | 19 / 19 | photometric overexposure streaming-temporal stress |
| `castle_symmetry` | 12 / 19 | registered-subset repetitive-structure streaming-temporal stress |
| `castle_dynamic_proxy` | 19 / 19 | dynamic-motion-boundary streaming-cache stress |

Blur, underexposure, clutter, KITTI and Horse remain excluded/deferred exactly as admitted by I5.8A.

## Native scientific status semantics

I5.8B does not invent a new scalar score or quality threshold.

A persisted I4.3D1 VDA summary may be:

- `pass`: all registered frames were processed and no provider-native degradation reason was emitted;
- `degraded`: provider execution completed, while the existing I4.3D1 semantics recorded rejected per-frame SfM alignments and/or fusion-capacity drops.

A generic runtime, integrity, environment or wrapper exception is **not** silently converted into a scientific `REJECTED` result. Such an exception preserves an `i5_8b_attempt_failure.json` artifact and terminates the campaign nonzero. Recovery, if ever necessary, must be designed explicitly from the preserved evidence rather than by retrying until a pass.

## Evidence retained per scenario

For every terminal provider execution, I5.8B records:

- complete registered-view processing count;
- accepted/rejected per-frame SfM alignment counts and fraction;
- native alignment rejection-code counts;
- descriptive accepted-frame relative-residual medians;
- canonical fused point count;
- fusion capacity drops;
- worker elapsed time;
- per-frame latency count, mean, median, p95, min and max;
- wrapper wall time;
- best-effort device-wide GPU-memory and system-wide RAM observations;
- exact runtime profile/checkpoint/source revision;
- hashes of provider summary, worker/input manifests, native depth NPZ and canonical NPZ/PLY when present;
- post-execution authoritative source/SfM integrity revalidation.

Resource observations and frame latencies are evidence, not qualification thresholds. Missing best-effort resource readings remain null.

## Historical evidence remains active

I5.8B preserves the I5.8A-bound historical VDA temporal anchors without rerunning or rewriting them:

- static sweep: `PASS`;
- motion: `DEGRADED`;
- low texture: `PASS`.

A successful current Castle campaign cannot silently erase the historical motion degradation. That interpretation belongs to I5.8C.

## Authority locks

I5.8B does not:

- rerun SfM or bundle adjustment;
- rerun MVS;
- rematerialize sources;
- use adaptive view reduction;
- introduce a permanent global view cap;
- relax alignment or fusion thresholds;
- modify VDA production defaults;
- retry until pass;
- compare/rank providers;
- promote model metric depth to physical metric scale;
- promote VDA estimated intrinsics to canonical camera authority;
- claim measurement or certified metrology authority;
- generate a surface-mesh authority claim;
- start MP-SfM, hybrid reconstruction, Auto policy, I5.9 or later roadmap phases.

## Modes

### `preflight`

Execution-free. Verifies the complete frozen foundation and real hardware-resolved VDA runtime. It writes `i5_8b_runtime_preflight.json` and must report `video_depth_anything_inference=False`.

### `production-native`

Executes the six admitted scenarios exactly once and sequentially. A successful campaign writes:

`i5_8b_video_depth_anything_robustness_replay.json`

A non-scientific execution/integrity failure preserves failure evidence and exits nonzero. Do not rerun such an attempt until its evidence has been inspected.

## I5.8B completion boundary

A completed replay does not itself declare universal VDA robustness. It authorizes only I5.8C, which consumes the frozen I5.8A + I5.8B evidence and defines the bounded qualification envelope while preserving historical limitations.
