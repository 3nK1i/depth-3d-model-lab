# I6.2A — One-Start Workflow Contract & Preflight

## Purpose

I6.2A defines the fail-closed admission contract for the future user-facing
**Start Reconstruction** action. It does not yet wire that contract into the
controller or UI; execution semantics remain unchanged until I6.2B.

## Existing architecture preserved

- `ReconstructionPage._start_runtime()` remains the current action entrypoint.
- `ReconstructionController.start_selected_offline()` remains the current
  selected-source launch helper.
- `OfflineReconstructionController` remains the isolated process owner.
- the existing application offline start guard remains the owner of cross-mode
  resource release/conflict handling and is required at launch commit.
- the offline worker remains the owner of hardware detection and adaptive
  pre-SfM frame planning.
- PyCOLMAP SfM remains the sole canonical camera/global-frame authority.

## New contract

`app/reconstruction_one_start_preflight.py` provides an immutable
`OfflineOneStartPreflight` snapshot and a non-executing builder that revalidates:

- selected workflow is `image_sequence` or `recorded_video`;
- selected source still exists and remains structurally valid;
- image sequences still contain at least three supported top-level images;
- recorded-video extension remains supported;
- no live/offline/DA3 reconstruction owner conflicts exist;
- isolated PyCOLMAP runtime still exists and is executable;
- the proposed session/output identity is a valid UUID-derived path;
- stale non-empty output reuse is rejected;
- output storage remains writable;
- free disk space is observed before future launch;
- application resource-guard rejection can be represented explicitly.

## Storage policy

I6.2A does **not** invent or retune a new storage threshold. It reuses the
existing `MVS_HOUSE_DEFAULT_MIN_FREE_DISK_GIB == 8.0` operational safety floor,
which is already part of the repository. This is a storage safety guard, not a
scientific reconstruction-quality threshold.

The preflight only observes storage. It never fills disk or creates an output
path while probing.

## Execution boundary

I6.2A authorizes only the currently existing offline execution envelope:

`source ingestion -> adaptive pre-SfM planning -> PyCOLMAP sparse SfM -> partial sparse result`

The contract explicitly records:

- execution scope: `sparse_sfm_only`;
- dense execution authorized: **NO**;
- persistent-scene promotion authorized: **NO**;
- production Auto qualified/authorized: **NO / NO**;
- certified metrology: **NO**;
- hardware adaptation owner: existing offline worker;
- canonical camera authority: PyCOLMAP SfM.

## Runtime transition contract

The legal transitions mirror the existing `OfflineReconstructionController`:

- `idle|completed|failed|cancelled -> starting`
- `starting -> running|failed`
- `running -> cancelling|completed|failed`
- `cancelling -> cancelled`

I6.2A does not change that state machine.

## Deferred to I6.2B

I6.2B will integrate this preflight into the existing controller launch path,
evaluate the existing application resource guard exactly at launch commit, and
perform the first qualified real UI reconstruction start. I6.2A itself performs
no worker launch and requires no manual UI validation.
