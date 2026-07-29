# EIA TECH — Depth & 3D Model Lab

A PyQt6 desktop application for live monocular depth estimation and
interactive 3D point-cloud visualization, built for Ubuntu 24.04 with
RTX 2060 / RTX 5090 GPUs as primary targets.

> **This is Phase 1 of an 8-phase build.** It delivers the complete UI
> shell only: layout, theming, navigation, and a demo runtime lifecycle
> for the buttons. **No capture, no inference, and no 3D rendering are
> implemented yet.** See [Status](#status--what-is-and-isnt-here) below
> before assuming any feature works.

## Status — what is (and isn't) here

| Area | State |
|---|---|
| Main window, header, mode tabs, splitters | ✅ Built |
| Dark industrial theme | ✅ Built |
| Source bar (Image/Video/Camera/RTSP buttons) | ✅ UI only - no capture wired |
| Runtime toolbar (Connect/Start/Pause/Stop) | ✅ UI only - driven by a **temporary local demo**, not a real state machine |
| Original source / Depth map panels | ✅ Chrome only - `set_frame()` exists but nothing calls it |
| 3D viewport | ✅ Chrome + tool column + render controls - canvas is a placeholder, no VisPy yet |
| Settings panel (Model/Depth/Render sections) | ✅ Built - no model registry behind it |
| Performance bar | ✅ Built - all values are placeholder dashes, nothing fabricated |
| Capture system, inference worker, state machine, model adapters, geometry pipeline | ❌ Not started - Phases 2-8 |

## Requirements

- Ubuntu 24.04 LTS (or any Linux with Qt6 platform plugin support)
- Python 3.11 or 3.12
- PyQt6 (see `pyproject.toml`)

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Running

```bash
./scripts/run.sh
# or
python main.py
```

The window enforces a minimum size of 1366×768 and lays out with
`QSplitter`s so every panel stays resizable and usable down to that
resolution.

## Tests

```bash
QT_QPA_PLATFORM=offscreen pytest tests/ui/ -v
```

`QT_QPA_PLATFORM=offscreen` lets the test suite run without a display
(e.g. in CI). Current coverage is scoped to what Phase 1 actually
claims: the window builds, unfinished nav modes stay disabled, minimum
resolution holds, and the demo connect/start/stop lifecycle drives
button enable/disable correctly, including the rule that **Stop never
disconnects the source**, which is the one lifecycle guarantee from the
full spec that's testable at this stage.

## Project layout

```
depth_3d_model_lab/
├── main.py                  # Entry point
├── app/
│   ├── main_window.py        # Assembles the shell, owns the Phase-1 demo lifecycle
│   ├── theme.py               # Palette + global QSS - no widget hardcodes a color
│   ├── settings.py             # QSettings wrapper (window geometry, splitters - no credentials)
│   └── widgets/
│       ├── header_widget.py
│       ├── mode_tabs.py
│       ├── source_bar.py
│       ├── runtime_toolbar.py
│       ├── original_view.py
│       ├── depth_view.py
│       ├── settings_panel.py
│       ├── performance_bar.py
│       ├── collapsible_section.py
│       ├── badge.py
│       └── viewport/
│           └── viewport_panel.py   # 3D viewport chrome; VisPy canvas lands in Phase 4
├── config/
│   └── app.yaml               # Not yet consumed by anything - see file header
├── core/                     # Empty - capture/inference/models/geometry land in later phases
└── tests/
    └── ui/
        └── test_main_window_shell.py
```

## What's next (Phase 2)

Source system: image/video/camera/RTSP capture with latest-frame
buffering, wired into `OriginalSourceView`, independent of any
inference state.

## Known limitations (Phase 1)

- Nothing renders real frames yet, display areas show static
  placeholder text.
- The 3D viewport has no VisPy canvas, rotate/pan/zoom do nothing.
- The runtime lifecycle is a local demo, not the real session-ID-based
  safe-stop system described in the full spec.
- Model, GPU-profile and render-setting controls in the settings panel
  are not backed by any registry, changing them has no effect.