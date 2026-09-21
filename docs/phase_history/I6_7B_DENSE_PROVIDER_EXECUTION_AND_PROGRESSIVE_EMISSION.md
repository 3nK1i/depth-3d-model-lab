# I6.7B — Dense Provider Execution & Progressive Emission

## Purpose

I6.7B is the execution/emission bridge for I6.7 Progressive Dense Visualization. It wraps one **already-selected concrete dense provider** and projects the provider's existing I4.3 progress events into the frozen I6.7A provisional dense-snapshot contract.

I6.7B does not select a provider, activate production Auto, admit a final dense result, persist a canonical dense scene, modify the Reconstruction controller/UI/viewport, generate a mesh, or establish metric/measurement/metrology authority.

## Execution boundary

The wrapper receives:

- an admitted `OfflineSfmResult` produced by authoritative PyCOLMAP SfM;
- one explicit `DenseProviderSelectionRequest`;
- one already-resolved `DenseProviderSelectionDecision` naming a concrete provider;
- the authoritative reconstruction-frame ID;
- a factory that constructs the existing I4.3 provider adapter with the I6.7B progress sink;
- a provisional snapshot sink.

The selected provider adapter still owns scientific execution. I6.7B does not rewrite provider algorithms, scientific thresholds, view-selection rules, alignment, fusion, or filtering.

All ten existing concrete provider identities are supported through the same wrapper:

- COLMAP MVS;
- Depth Anything 3;
- RollingDepth;
- Video Depth Anything;
- Metric3D v2;
- UniDepthV2;
- MoGe-2;
- Depth Pro;
- Depth Anything V2;
- Marigold.

`Auto` remains policy-only and cannot execute as a provider.

## Progressive status emission

Existing `DenseProviderProgressEvent` values are translated into `ProgressiveDenseSnapshot` values. I6.7B uses a visualization-only status throttle with a target of 24 status buckets plus stage transitions. The throttle reduces redundant preview traffic only; it never changes provider work.

Revisions strictly increase. Gaps remain valid when a snapshot sink fails or a later transport stage coalesces superseded snapshots. Successfully emitted snapshots are transition-validated against the frozen I6.7A contract, so provider/stage/frame identity drift and stage regression fail closed.

Snapshot-sink failures are fail-soft: a visualization sink cannot invalidate an otherwise valid dense provider result. Failed emissions are recorded and are not retried until they pass.

## Bounded actual geometry preview

I6.7B defines a visualization-only preview maximum of **4096 points**. This is not a reconstruction point cap and is not a quality threshold.

The preview loader supports the canonical dense artifacts already produced by the provider architecture:

- `.ply` dense point clouds, including ASCII and scalar binary PLY vertex payloads;
- `.npz` dense geometry with `points_reconstruction` and optional RGB/confidence/view-support arrays.

Sampling is deterministic and preserves the first/last population positions when reduction is required. The full dense artifact remains unchanged.

After a successful or degraded provider result is constructed, I6.7B emits one final bounded **provisional** geometry snapshot from `canonical_geometry_ref` before returning the result to any later admission/persistence layer. The result's `final_preview_revision` is updated only to identify that preview revision.

This preview is still explicitly not final authority.

## Authority boundary

Every I6.7B snapshot remains:

- provisional visualization only;
- bound to one admitted sparse-result digest;
- bound to one dense-run/request fingerprint;
- bound to the authoritative PyCOLMAP reconstruction frame.

I6.7B cannot claim or create:

- final dense-result admission;
- persistent dense-scene creation;
- canonical geometry authority for the progressive path;
- camera/global-frame authority;
- physical metric scale;
- measurement authority;
- certified metrology;
- surface mesh generation;
- provider promotion.

PyCOLMAP SfM remains the sole camera/global-frame authority.

## Failure semantics

The following are fail-closed contract violations:

- selected adapter identity not matching the resolved provider decision;
- progress events changing provider/implementation identity;
- progress events reporting a view population outside the admitted request;
- emitted I6.7A transition identity/stage regression;
- a successful/degraded final result changing reconstruction frame.

The following preview-only failures are fail-soft and recorded:

- provisional geometry preview cannot be parsed;
- final geometry preview cannot be parsed;
- downstream provisional snapshot sink throws.

I6.7B does not retry-until-pass and does not weaken scientific settings to obtain evidence.

## Real qualification path

Before freezing I6.7B, execute the existing qualified COLMAP-MVS provider on one already-admitted real sparse reconstruction through `scripts/run_i6_7b_progressive_dense_emission_smoke.py`.

The smoke must prove:

- real provider execution completes or is explicitly degraded under the existing provider contract;
- one or more progressive dense snapshots are emitted;
- at least one actual dense geometry preview is emitted;
- preview point count is at most 4096;
- the provider's canonical dense geometry artifact exists;
- every snapshot validates against I6.7A;
- the progressive path does not admit/persist/promote the dense result;
- production Auto remains inactive;
- PyCOLMAP authority and arbitrary-scale semantics remain intact.

Real provider negative/degraded evidence must be preserved rather than hidden or retried until a passing configuration is found.

## Scope exclusions

I6.7B adds no:

- application/controller dense transport, coalescing, or memory policy (I6.7C);
- incremental dense viewport rendering (I6.7D);
- sparse-to-dense admitted-result handoff/persistence (I6.7E);
- professional dense UX/performance closure (I6.7F);
- mesh reconstruction (I7);
- Gaussian Splatting (I8);
- calibration or physical metric-scale claim (I9).

## I6.7B fusion execution-policy correction

The first real 24-view COLMAP-MVS I6.7B qualification preserved all scientific settings and completed CUDA PatchMatch for every selected reference, but Stereo Fusion entered COLMAP's cached-workspace path and did not complete within the qualification window. The preserved negative evidence showed `use_cache: 1`, a single effective fusion thread, no final worker result, and therefore no actual dense geometry preview.

A fusion-only diagnostic reused the exact frozen PatchMatch workspace and changed only the Stereo Fusion execution policy. With `use_cache=False` and `num_threads=-1`, COLMAP loaded the workspace using the host's six effective CPU threads and fused all 24 views in about 39 seconds, producing 1,686,154 points. No scientific fusion threshold, image-size setting, sparse camera, depth/normal map, provider-selection rule, or authority boundary changed.

The production COLMAP-MVS worker therefore uses:

- `StereoFusionOptions.use_cache = False`;
- `StereoFusionOptions.num_threads = -1` (COLMAP/PyCOLMAP resolves effective host parallelism);
- the existing configured cache-size value remains recorded for provenance but does not enable CachedWorkspace;
- all Stereo Fusion scientific defaults remain unchanged;
- `max_image_size` remains the existing configured value.

The worker records this execution policy in its result as `stereo_fusion_execution_policy` with `scientific_fusion_parameters_changed=False`.

This is a narrow I6.7B execution-performance correction, not the full hardware-adaptation policy planned for I6.11. The real qualification must be rerun from a clean output directory after this correction; the original timed-out evidence remains preserved as negative evidence.

The real smoke also now checks provider completion/failure status before asserting the presence of geometry, so a timeout or provider failure is reported as the primary cause instead of the secondary symptom `no actual dense geometry preview`.

## I6.7B terminal worker-stage ordering correction

The corrected real COLMAP-MVS qualification completed Stereo Fusion and wrote the dense `fused.ply`, but progressive validation then rejected a stage regression. Read-only source inspection demonstrated that the external worker records the terminal state `complete` after `stereo_fusion`, while the MVS adapter had no explicit `complete` mapping. The generic fallback therefore translated that terminal worker state back to `DenseProviderProgressStage.PROVIDER_EXECUTION` after `FUSION_FILTERING`, and the frozen I6.7A transition validator correctly failed closed.

A production-faithful integration regression now publishes the real terminal sequence `stereo_fusion -> complete`. Before the correction it reproduces `ReconstructionValidationError: Progressive dense execution stage cannot regress within one run.` The narrow production correction maps worker `complete` explicitly to `DenseProviderProgressStage.FINALIZATION` with provisional progress `0.95`; the existing canonical MVS geometry-ready finalization remains responsible for the later completed result/final preview path.

The stage-regression invariant is unchanged. No arbitrary regression is accepted, no worker scientific setting is changed, no provider ranking or Auto policy is changed, and no I6.7C transport/backpressure behavior is introduced.


## Final real qualification and closure readiness

The final clean production-path qualification used the uniquely recovered frozen PyCOLMAP SfM result:

- source report: `i6_6b_worker_progressive_emission_20260917_144852/real_pycolmap_progressive/host_run/real_sfm_report.json`;
- source SHA256: `2fba39d00b10043666f31b598beab7c1892ef127daf5da73e6104eb84f347228`;
- reconstruction frame: `i6_7b:pycolmap_sfm:23deb9ac-c7b8-4ec0-a9f1-761a1f36215e`;
- registered sparse views: 24.

The final real qualification evidence is:

`i6_7b_dense_provider_progressive_emission_final_20260918_172610`

The production run completed undistortion, CUDA PatchMatch, geometric consistency, and Stereo Fusion. COLMAP fused all 24 views, produced **1,686,160** dense points, and wrote the canonical `fused.ply` artifact. The progressive wrapper emitted 23 validated revisions, including one actual bounded geometry snapshot with 4096 preview points.

The observed stage sequence was monotonic and ended in finalization:

`preparing -> view_selection -> provider_execution ... -> fusion_filtering -> finalization -> finalization -> finalization`

The previous invalid `fusion_filtering -> provider_execution` regression did not recur. The worker terminal state now advances through `FINALIZATION` while the frozen I6.7A stage-regression invariant remains unchanged and fail-closed.

Final real qualification results:

- `I6_7B_REAL_PROGRESSIVE_DENSE_EMISSION=PASS`;
- `FINAL_PROVIDER_STATUS=completed`;
- `I6_7B_FINAL_REAL_RC=0`;
- progressive final admission: **NO**;
- persistent dense-scene creation: **NO**;
- production Auto activation: **NO**;
- PyCOLMAP camera/global-frame authority preserved: **YES**;
- metric-scale claim: **NO**;
- surface mesh generation: **NO**.

The final automated closure before the real run also passed with:

- 32 focused I6.7B tests;
- 27 frozen I6.7A regressions;
- 179 concrete-provider predecessor tests;
- 21 I6.4 provider-boundary tests;
- 35 I6.6 authority-boundary tests;
- 4583 full unit tests;
- compile, shell-syntax, scope, and diff-hygiene checks all returning zero.

No retry-until-pass policy, scientific threshold, provider ranking, production Auto behavior, controller/UI/viewport behavior, persistence behavior, final admission rule, or future I6.7C transport/backpressure behavior was introduced to obtain this result. The earlier failed and degraded evidence remains preserved.

I6.7B is therefore **qualified for immutable closure freeze**. It becomes the closed/frozen predecessor for I6.7C only after the post-I6.7B source checkpoint and its SHA256 sidecar are created and independently verified. The closure receipt and checkpoint are operational handoff artifacts and are not additional I6.7B product behavior.
