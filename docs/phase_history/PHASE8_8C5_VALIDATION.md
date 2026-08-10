# Phase 8.8C5 — Live Tracking / Map Diagnostics

## Scope

C5 is presentation-only. It adds a compact top-left Reconstruction viewport HUD
using telemetry already present in `LiveSlamObservation`:

- authoritative tracking state (`initializing`, `tracking`, `recovering`, `lost`, ...),
- real per-frame Stella processing latency with bounded rolling median / p95,
- live map revision, keyframe count and landmark count,
- map growth between actual map revisions,
- native feature/inlier diagnostics only when the backend really supplies them.

The current qualified Stella worker reports feature counts as zero, so C5 must
show **Not reported by Stella** rather than inventing a feature-quality score.

## Safety / architecture

- No `core/reconstruction` contract or protocol change.
- No native Stella worker change or rebuild required.
- No metric scale, world alignment, confidence, or metrology claim is introduced.
- Diagnostics are UI presentation only and cannot start/stop/modify SLAM.
- Rolling latency history is bounded to 120 samples.
- The HUD toggle is enabled by default, but the overlay is shown only in live context; offline/idle workspaces stay uncluttered.
- A new live session clears diagnostics history but preserves the user's HUD visibility preference.
- Stop/failure retains the last diagnostic sample and labels it explicitly as `LAST • ...`.

## Validation gates

1. `./scripts/run_phase8_8c5_tests.sh quick`
2. `./scripts/run_phase8_8c5_tests.sh protocol`
3. `./scripts/run_phase8_8c5_tests.sh backward`
4. `python scripts/run_phase8_8c5_diagnostics_preview.py`
5. `python scripts/run_phase8_8c5_live_diagnostics_smoke.py --max-frames 500 --playback-speed 2.0`
6. Normal `python main.py` sanity.

The existing Phase 8.8C4 Stella worker is intentionally reused. A C5 validation
must not require rebuilding or changing the qualified worker.
