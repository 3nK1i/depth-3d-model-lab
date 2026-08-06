# Phase 7.5 Step 10 — Depth Color-Bar Qt Crash Fix

## Scope

Focused pre-commit correction for the native Qt segmentation fault reported during the complete UI test group.

Observed native stack:

- `app/widgets/depth_view.py::_ColorBar.paintEvent`
- Qt backing-store event processing through pytest-qt

The restart state-machine tests themselves completed; the crash occurred later while Qt processed a queued custom paint event in the long-lived UI test process.

## Correction

- Removed the Python-managed `QPainter` override from `_ColorBar`.
- Replaced it with an equivalent vertical `qlineargradient` stylesheet.
- Enabled `WA_StyledBackground` so the gradient is rendered through Qt's standard widget style path.
- Preserved all existing colormaps, stop order, rounded corners, size, and dynamic updates.
- Extended the existing media-toolbar regression test to verify that the selected colormap updates the stylesheet and includes its first and last color stops.

This avoids a manually managed painter during backing-store teardown and does not change inference, restart, depth rendering, measurements, or state-machine behavior.

## Validation in delivery environment

- Python compilation: passed
- Static check that `_ColorBar` no longer defines `paintEvent`: passed
- Static check that the stylesheet contains `qlineargradient`: passed

PyQt6 is not installed in the delivery environment. The focused and complete Qt suites must be rerun in the configured `depth` environment.

## Target validation

```bash
pytest -q \
  tests/ui/test_media_toolbars.py \
  tests/ui/test_state_machine.py \
  tests/ui/test_main_window_shell.py

./scripts/run_phase7_5_step10_tests.sh full
```

Expected: no native segmentation fault in `depth_view.py` and all UI tests pass.
