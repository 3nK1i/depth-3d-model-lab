# I4.2E.2a — Native Visibility Coverage Contract Correction

## Trigger

After I4.2E.2 was correctly installed, its dedicated tests passed but the full
I4.2E suite exposed one backward-contract regression. The native visibility
parser rejected a valid synthetic sidecar whose surviving fused points referenced
only the MVS image-index subset `{2, 3, 5}` of an expected six-view campaign.

The parser had incorrectly promoted *global full-view coverage* into a native
`fused.ply.vis` validity requirement.

## Correction

I4.2E.2a preserves the native COLMAP visibility format and all fail-closed
structural checks, but distinguishes two concepts:

- **campaign participation** — established upstream by I4.2D (47 reference views);
- **surviving fused-point support** — the subset of MVS image indices that occurs
  in `fused.ply.vis`.

A valid sidecar may contain any non-empty global subset of `[0, 46]`. Every
fused point must still have non-empty support, every per-point support set must
be duplicate-free, and every observed index must remain in range. Point-count
alignment, truncation, trailing-byte, hash, and immutability checks remain
unchanged.

The worker now persists the observed index set/count and whether it happens to
cover all 47 views as diagnostics. Neither full-set coverage nor a unique-view
count of 47 is used as an acceptance gate.

## Authority locks

No SfM, BA, PatchMatch, Stereo Fusion, pose, intrinsics, production `d12_sparse`
policy, provider authority, metric scale, measurement authority, metrology, mesh,
or I4.3 authorization changes are introduced. Existing persistent I4.2A-D
evidence remains valid and must not be regenerated for this correction.

## Validation intent

The correction is accepted only if:

1. the dedicated I4.2E.2a focused/stress suite passes;
2. the pre-existing I4.2E.2 dedicated suite still passes;
3. the complete I4.2A-E test stack passes with no regression;
4. I4.1.9 backward sanity remains passing;
5. `git diff --check` is clean.

Only then may a fresh real I4.2E qualification attempt be launched against the
persistent I4.2D `fused.ply` + `fused.ply.vis` pair.

## Post-overlay test-contract consistency correction

Full-repository focused validation exposed one stale pre-correction unit test,
`test_i42e_track_evidence_requires_all_47_images`, that still expected the
removed global full-view-coverage gate. The implementation was already using
the corrected native-visibility contract; this stale test was contradictory.

The stale test is replaced with an acceptance test for a valid visibility
subset, and the I4.2E.2a regression suite now explicitly scans the historical
I4.2E unit contract to prevent reintroduction of the obsolete exact-global-set
expectation. No runtime, MVS, filtering, threshold, authority, or production
policy behavior changes in this correction.
