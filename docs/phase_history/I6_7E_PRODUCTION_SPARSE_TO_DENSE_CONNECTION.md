# I6.7E — Explicit post-sparse production dense connection

## Goal and boundaries

This is the completion of the **application-side** sparse → dense connection; it
is not I6.7F full-point-cloud rendering or a new scientific algorithm.

After **One Start** has produced and the application admitted its *own* PyCOLMAP
sparse result, the separate **Build dense (COLMAP MVS)** button can launch one
explicit concrete provider for that same session. It is never launched
implicitly by sparse completion or by Auto. A 4,096-point I6.7C/D bounded
provisional preview may be displayed during processing but cannot become a
complete result or persistent scene.

The GUI owns a QProcess in an isolated Linux process group. Nested PyCOLMAP,
PatchMatch and Stereo Fusion children inherit that group; Cancel dense and
shutdown signal the **group**, not only the Qt-owned parent. The worker emits
only a final-admission *claim* after the E2A candidate/E2B1 admission gates.
A second isolated process reconstructs the typed admission from its on-disk
manifest, binds it to the current sparse session/run/provider/fingerprint and
rechecks the complete PLY SHA256, count, RGB, and bounds. Only a successful
independent verification permits the application's **admitted** signal and
terminal success. No full-PLY read/hashing runs in the Qt thread.

The added runtime row distinguishes *full dense artifact admitted* from the
**full GPU viewer pending I6.7F**. Existing canonical camera/global-frame
owner remains PyCOLMAP; raw gauge stays arbitrary. No canonical surface mesh,
measurement authority, certified metrology, production Auto, or persistent
scene is created. The frozen A–D contracts, I6.7D GPU canvas, E1/E2A/E2B1
admission machinery, and E2B2A external provider worker are unchanged.

The I6.7E app coordinator and reconstruction page must necessarily change to
connect the authorized worker and display its explicit action; they are
intentionally modified *after* the immutable I6.7D checkpoint, which remains
preserved. Legacy I6.7D snapshot preview, projection and bounded transport
still use their unchanged interfaces.

## Workstation qualification

1. Verify patch ZIP SHA256 and exact pre-apply source hashes. Apply only on
   matching state. Preserve the full frozen I6.7D checkpoint separately.
2. Run `bash scripts/run_i6_7e_production_tests.sh focused` — requires real
   PyQt6/pytest-qt, headless viewport test double and Linux process-group test.
   The **fake nested MVS** cancellation test must show that no child survives.
3. Run `bash scripts/run_i6_7e_production_tests.sh closure`: frozen-source,
   predecessor, architecture, whole-unit and disk-safety gates; no actual SfM/MVS.
4. Manually verify initial button disabled, completion enables one *explicit*
   dense action, live provisional label/non-metric warning, Cancel dense
   responsiveness and no new authoritative scene. Use only a freshly admitted
   One Start sparse session, never relabel the frozen I6.7B demo data as a new run.
5. **Real MVS qualification is separate, explicitly authorized and expensive.**
   Confirm source/data/disk/GPU resources and cancellation gates before its
   first and only intended run. Preserve failures without retry-until-pass.

I6.7E is not declared closed or frozen just because a ZIP installs or synthetic
UI tests pass. I6.7F remains the full-density GPU rendering/performance/UX
and complete I6.7 phase-closure milestone.
