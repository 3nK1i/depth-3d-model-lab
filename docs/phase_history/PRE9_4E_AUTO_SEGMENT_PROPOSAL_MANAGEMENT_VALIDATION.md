# Pre-9.4E Auto Segment Proposal Management

Purpose: close the proposal-cleanup UX gap before Interactive Measurement and the 9.4E Gaussian viewer.

## Scope

- Keep **Reject** as a non-destructive review state.
- Add a compact `×` delete affordance to every Auto Segment proposal row.
- Allow permanent removal of Pending or Rejected proposal-backed objects.
- Persist deletion immediately through the existing semantic-workspace store so exact-source restore does not recreate the deleted proposal.
- Clear selected/prompt/merge state safely when the deleted proposal was active.
- Protect Accepted objects and merged semantic results from destructive deletion.
- Preserve existing merge selection, prompt refinement, class assignment, persistence schema, and semantic authority boundaries.
- Do not rerun SAM2 during deletion. A later explicit Auto Segment generation may legitimately create a new similar proposal.

## Authority

Semantic masks/classes remain context only:

- geometry authority: false
- metric authority: false
- measurement authority: false

No canonical scene, Gaussian, reconstruction, calibration, or measurement contract is modified by this completion step.

## Validation gates

```bash
./scripts/run_pre9_4e_auto_segment_proposal_tests.sh quick
./scripts/run_pre9_4e_auto_segment_proposal_tests.sh integration
./scripts/run_pre9_4e_auto_segment_proposal_tests.sh backward
python scripts/run_pre9_4e_auto_segment_deletion_smoke.py
```

Manual validation is limited to the affected UX:

1. Open Measure & Inspect -> Auto Segment on an image with saved proposals.
2. Reject an unwanted proposal and click its right-side `×`.
3. Confirm the mask and row disappear immediately.
4. Close Auto Segment, reopen the same project/image, and confirm the deleted proposal stays gone.
5. Confirm Accepted and merged semantic results show a disabled/protected delete control.
6. Confirm merge selection still works and explicit Auto Segment regeneration may create new proposals.
