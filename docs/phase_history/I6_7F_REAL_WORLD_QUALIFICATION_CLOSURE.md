# I6.7F real-world qualification closure — 2026-09-21

## TempleRing — completed dense viewer qualification

An independently admitted COLMAP MVS RGB fused PLY containing 163,827
points was displayed in full by the GPU-backed viewer. Its preserved
artifact passed SHA256 verification.

The successful rendering evidence does not establish that a persistent
production scene or sustained frame-rate target has been qualified.

## South Building — incomplete large-scene qualification

The accepted sparse reconstruction registered 87 views. PatchMatch
produced the corresponding geometric depth and normal maps.

Full-resolution stereo fusion did not complete on the 15 GiB RAM
workstation. The original attempt was terminated by the system after
memory exhaustion; the bounded recovery attempts did not produce an
admitted final dense artifact. The last cached, one-thread recovery
was deliberately stopped after remaining on image 1/87 for approximately
40 minutes.

The approximately 20 GiB session was moved to:
 /home/eiatech/Desktop/reconstruction_evidence_preserved/03d5f129-781f-419f-90aa-95673a5306fc

The preserved session's historical request paths still refer to its
former /tmp location. They must be revalidated before future recovery.

South Building is not counted as a successful dense reconstruction.
No additional local fusion retry is planned; a future higher-RAM host
may reuse the preserved stereo workspace following integrity and
environment checks.

## Automated checks

The I6.7F closure command reported 4,661 passing full-suite tests,
26 focused tests, 124 predecessor tests, and passing scope, compile,
shell, storage, and diff checks. The two fusion-recovery test files
reported 25 passing tests.

These results do not replace final manual main-application UI acceptance
or establish a successful South Building dense admission.

## Final manual GPU-viewer acceptance — 2026-09-21

Result: PASS for the dedicated I6.7F complete-cloud viewer.

The original, independently admitted TempleRing session was reopened
without rerunning SfM or MVS.

- Verified points: 163,827
- Displayed points: 163,827
- Geometry submissions: 1
- CPU preparation: 1292.326 ms
- Qt marker submission: 13.858 ms

The operator visually confirmed the complete RGB cloud and successfully
tested orbit, pan, zoom, and Fit complete cloud. The fit control was
confirmed by deliberately moving away from the fitted view and returning.

Qualification boundaries:
- Sustained rendering FPS was not measured.
- No persistent scene was created.
- No metric scale was claimed.
- Reopening an existing admitted cloud in the main Reconstruction page
  remains unqualified.
- The preserved TempleRing copy cannot currently be replayed from its
  relocated path without resolving the original sparse-SfM path binding.
- South Building did not produce an admitted final dense artifact.

These limitations do not invalidate the successful dedicated full-cloud
viewer acceptance.
