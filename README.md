# EIA TECH - Depth & 3D Model Lab

A PyQt6 desktop application for live monocular depth estimation and interactive 3D point-cloud visualization on Ubuntu 24.04.

## Current status

Phases 1-4 are implemented:

- PyQt6 application shell and dark industrial UI
- Image, video, camera and RTSP capture
- Process-isolated inference lifecycle with session-based stale-result rejection
- Development mock model
- Depth colorization and point-cloud generation
- Interactive VisPy 3D viewport
- Runtime settings and performance footer

Phase 5 adds the first real model: MoGe-2, hardware-aware profiles and real GPU metrics.

## Requirements

- Ubuntu 24.04 LTS
- Python 3.11 or 3.12
- NVIDIA driver and CUDA-compatible PyTorch for real GPU models

## Installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

MoGe-2 dependencies will be installed separately from the base UI/runtime environment during Phase 5.

## Run

```bash
./scripts/run.sh
# or
python main.py
```

## Tests

```bash
QT_QPA_PLATFORM=offscreen pytest -q
```

## Project structure

```text
app/                 Qt UI and application controllers
core/capture/        Image, video, camera and RTSP sources
core/inference/      Worker process and IPC manager
core/models/         Unified model adapter interface
core/geometry/       Depth, point-cloud, filtering and export utilities
core/runtime/        State machine, sessions and watchdog
tests/               UI and integration tests
```
