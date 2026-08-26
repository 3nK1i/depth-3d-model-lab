# I4.3B — COLMAP MVS Production Provider Adapter

## Scope

I4.3B binds the already-qualified COLMAP/PyCOLMAP MVS methodology behind the I4.3A canonical dense-provider contract. It is intentionally narrow: no UI/controller wiring, no Auto policy, no DA3/temporal/single-image provider work, no adaptive view reduction, and no provider promotion.

The provider remains a **candidate** until the controlled cross-provider operating-envelope qualification in I4.3G and architecture closure in I4.3H. I4.3B therefore establishes an executable production-shaped adapter, not a universal-winner claim.

## Locked authority

- PyCOLMAP offline SfM remains the sole camera/global-frame authority.
- I4.3B consumes an existing `OfflineSfmResult`; it never reruns SfM, BA, or pose refinement.
- MVS dense geometry is generated directly in the authoritative SfM frame.
- All authoritative registered SfM views are used in I4.3B. Adaptive view selection is deferred to I4.3F.
- There is no permanent global frame/view cap.
- Geometric consistency remains enabled; the validated MVS quality path is not weakened.
- Input staged images and authoritative sparse binaries are SHA-256 protected before/after execution.
- `fused.ply` and native `fused.ply.vis` must have matching point counts.
- No production filtering policy change is made here. Downstream filtering/persistence stays unchanged.
- No metric scale, measurement authority, certified metrology, surface mesh, or provider promotion is established.

## Execution architecture

The main `depth` process owns the provider adapter and I4.3A contracts. Heavy PyCOLMAP MVS remains in the isolated PyCOLMAP interpreter through `reconstruction_pycolmap_dense_provider_mvs_worker.py`.

The worker generalizes the exact MVS sequence already exercised in the House positive control and TempleRing qualification:

1. consume the selected authoritative PyCOLMAP sparse model and registered staged images;
2. undistort only the registered views;
3. verify that the dense COLMAP workspace preserves the source camera poses;
4. run PatchMatch Stereo with geometric consistency enabled;
5. require geometric depth/normal maps for every selected reference view;
6. run geometric Stereo Fusion;
7. validate `fused.ply` + `fused.ply.vis` correspondence;
8. re-hash source images and sparse binaries unchanged.

## Progressive-preview compatibility

I4.3B implements the UI-neutral I4.3A progress surface. It emits preparing/view-selection/provider-execution/fusion/finalization progress and polls the number of completed geometric depth maps plus worker resource snapshots.

I4.3B does **not** invent provisional fused geometry while PatchMatch is incomplete. The v1.0 requirement for progressively visible reconstruction remains locked; actual viewport transport/provisional geometry policy is productized later in I6. Final provider geometry remains separate from provisional preview geometry.

## Real-evidence smoke policy

I4.2 already performed expensive real PatchMatch/Stereo Fusion runs on House/TempleRing and froze the MVS evidence. The I4.3B binding smoke therefore revalidates that frozen evidence, its PLY/VIS hashes/correspondence, I4.2F authority, I4.3A authorization, and the current isolated PyCOLMAP MVS runtime **without rerunning heavy MVS merely for repetition**.

The generic adapter/worker execution contract is covered by unit/integration/stress tests. Real cross-provider production operating-envelope execution is intentionally retained for I4.3G.

## Exit gate

I4.3B may close only when:

- focused, stress, complete I4.2 + I4.3A/B tests pass;
- isolated production-worker import probe passes;
- I4.1.9 backward sanity remains green;
- `git diff --check` is clean;
- the real I4.2F/I4.3A binding smoke passes against the current isolated PyCOLMAP MVS runtime;
- provider qualification remains `candidate`, with I4.3G/I4.3H still required.

On PASS, I4.3C — DA3 Production Provider Adapter — is next.
