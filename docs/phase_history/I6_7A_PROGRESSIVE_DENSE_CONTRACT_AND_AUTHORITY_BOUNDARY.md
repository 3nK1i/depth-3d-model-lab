# I6.7A — Progressive Dense Contract & Authority Boundary

## Purpose

I6.7A is the contract-only foundation for I6.7 Progressive Dense Visualization. It does not execute a dense provider, select a provider, activate production Auto, modify reconstruction thresholds, transport a dense preview, or change the Reconstruction UI.

The step extends the already-frozen I4.3 dense-provider progress surface with a concrete bounded dense-point preview contract that later I6.7 steps can emit, transport, coalesce, render, and finally supersede with an admitted dense result.

## Authority boundary

A `ProgressiveDenseSnapshot` is **provisional visualization only**. It is bound to:

- one admitted sparse SfM session;
- one dense execution/run identity;
- one concrete canonical provider implementation;
- one dense request fingerprint;
- one digest of the admitted sparse result used as dense input;
- one reconstruction frame owned by authoritative PyCOLMAP SfM.

It cannot claim:

- final dense-result admission;
- persistent canonical dense-scene creation;
- canonical geometry authority;
- camera/global-frame authority;
- metric scale;
- measurement authority;
- certified metrology;
- a generated surface mesh;
- provider promotion.

These claims fail closed during object construction and payload parsing.

## Provider neutrality

The contract accepts all ten canonical concrete provider implementations already frozen by I4.3/I6.4:

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

`Auto` is not a concrete executable provider and cannot appear as `provider_id` in a progressive dense snapshot. I6.7A does not modify provider availability, qualification, recommendation, manual selection, or the fail-closed production Auto gate.

## Dense preview

`ProgressiveDenseGeometryPreview` carries an optional bounded sample of dense points. Each point can contain:

- XYZ in the reconstruction frame;
- optional RGB;
- optional provider/fusion confidence;
- optional multi-view support.

The sample index is preview-local only. It is not persistent geometry identity.

I6.7A deliberately defines **no operational preview-size cap**. I6.7B/C own bounded emission and memory/backpressure policy. A visualization limit must never become a scientific dense-reconstruction threshold.

## Statistics and transitions

Each snapshot records input/selected/processed view counts, current provisional dense-point count, and optional confidence/support summaries.

Transitions require stable run/source/provider/frame identities and strictly increasing revisions. Revision gaps are legal so future I6.7C coalescing can discard superseded intermediate snapshots. Execution stages cannot regress.

Dense-point counts are deliberately **not required to increase monotonically**. Alignment, fusion, confidence filtering, duplicate removal, and outlier cleanup may legitimately reduce provisional point count.

## Existing contract relationship

I6.7A does not replace `DenseProviderProgressEvent`. The older I4.3 event remains the provider-neutral execution/status contract. I6.7A adds the stronger scene/session/provenance/geometry authority envelope needed for product-level progressive dense visualization.

The existing persistent Dense RGB scene contract remains the later admitted/persistent authority boundary. A progressive snapshot cannot silently become that persistent scene.

## Scope exclusions

I6.7A adds no:

- provider execution or worker emission (I6.7B);
- transport, backpressure, or memory policy (I6.7C);
- incremental dense viewport rendering (I6.7D);
- sparse-to-dense final handoff/admission (I6.7E);
- professional dense UX/performance closure (I6.7F);
- mesh reconstruction (I7);
- Gaussian Splatting (I8);
- calibration or physical metric-scale claim (I9).

PyCOLMAP SfM remains the sole camera/global reconstruction-frame authority.
