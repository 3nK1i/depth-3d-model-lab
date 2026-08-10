# Phase 8.8B4 — Status / Progress Foundation

## Scope

B4 is presentation-only. It consumes the existing authoritative Reconstruction
controller signals and worker events; it does not introduce a second lifecycle
state machine or change backend execution.

The dormant Reconstruction page now presents:

- live/offline authoritative state;
- current user-facing stage;
- stage-local offline progress when a backend supplies a valid fraction;
- indeterminate activity for live tracking and other non-finite work;
- elapsed runtime;
- compact live statistics (processed frames, observations, poses, keyframes,
  landmarks, mean latency when available);
- compact offline SfM result statistics;
- user-facing failure/cancellation messages without dumping raw backend logs.

## Authority rules

- Live progress is never represented as a fabricated percentage.
- Offline percentages are stage-local values emitted by the existing worker.
- The current A3 production worker exposes source ingestion and camera
  reconstruction only; B4 does not pretend dense geometry or neural stages ran.
- Estimated/Test Mode remains non-metric and not certified for metrology.
- `offline_log_received` is deliberately not bound to the main page.
- Reconstruction navigation remains disabled until B5.


## Terminal presentation precedence

A live source may become disconnected immediately after a clean live-SLAM stop.
B4 preserves the authoritative terminal runtime presentation (`Stopped` / failure)
when that post-stop source-readiness refresh arrives. Source readiness still updates
independently, but it may not overwrite the completed runtime outcome. Selecting a
different offline workflow continues to present its own readiness normally.

## No architecture changes

- no `core/reconstruction` changes;
- no new capture owner;
- no PyCOLMAP/Stella/DA3/Gaussian/NeRF import in production `depth`;
- no backend start from source selection;
- no MainWindow routing change;
- no Reconstruction navigation enablement.

## Validation

```bash
./scripts/run_phase8_8b4_tests.sh quick
./scripts/run_phase8_8b4_tests.sh backward
```

Real offline status/progress path:

```bash
python scripts/run_phase8_8b4_offline_status_ui_smoke.py \
  --images /home/eiatech/Desktop/phase8_3c_south_building/south-building/images \
  --sfm-python ~/anaconda3/envs/eia-pycolmap-probe/bin/python
```

Real live status/telemetry path:

```bash
python scripts/run_phase8_8b4_live_status_ui_smoke.py \
  --max-frames 500 \
  --playback-speed 2.0
```

Manual UI preview remains available through:

```bash
python scripts/run_phase8_8b1_reconstruction_page_preview.py
```

At B4 completion, Reconstruction must still be disabled in normal navigation.
