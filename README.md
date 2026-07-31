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

Phase 5 is in progress. Step 1 is implemented:

- NVIDIA GPU discovery without creating model tensors in the Qt process
- PyTorch/CUDA build checks and clear CPU fallbacks
- Capability-based Efficient, Balanced and Ultra profiles
- Automatic profile selection for RTX 2060-, RTX 5090- and other NVIDIA-class GPUs
- Persistent profile selection in the MODEL card

The MoGe-2 adapter, worker-side warm-up/inference and live GPU metrics follow in the next steps.

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

## Hardware check

Run the standalone detector before installing or loading MoGe-2:

```bash
python scripts/check_hardware.py
```

The detector uses `nvidia-smi` first and reads PyTorch build metadata when available. A CPU-only PyTorch installation is reported explicitly instead of silently selecting CUDA. Profile thresholds are capability/VRAM based, not hard-coded to one GPU product name.

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
