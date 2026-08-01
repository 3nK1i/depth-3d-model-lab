# EIA TECH - Depth & 3D Model Lab

A PyQt6 desktop application for live monocular geometry estimation and interactive 3D point-cloud visualization on Ubuntu 24.04.

## Current status

Phases 1-4 and Phase 5 Steps 1-2 are implemented:

- Image, video, camera and RTSP capture
- Process-isolated inference worker with session-based stale-result rejection
- Latest-frame live-stream backpressure
- Hardware-aware Efficient, Balanced and Ultra profiles
- Development mock adapter for lifecycle diagnostics
- MoGe-2 metric depth, direct point maps, normals, mask and intrinsics
- Interactive VisPy point-cloud viewport
- Measured latency, FPS, point count and worker-side VRAM metrics

## Requirements

- Ubuntu 24.04 LTS
- Python 3.11 or 3.12
- NVIDIA driver and CUDA-compatible PyTorch for GPU inference

## Base installation

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## MoGe-2 installation

Install the official model package separately so the base UI remains usable without model dependencies:

```bash
python -m pip install -r requirements/models-moge2.txt
```

The first MoGe-2 start downloads the selected checkpoint from Hugging Face. The automatic RTX 2060 profile selects `Ruicheng/moge-2-vits-normal`; larger GPUs resolve to the Balanced or Ultra variant.

## Hardware check

```bash
python scripts/check_hardware.py
```

## Run

```bash
./scripts/run.sh
# or
python main.py
```

Select a source, choose **MoGe-2** in the MODEL card, connect the source, then press **Start Inference**. The optional **Development Mock** remains available when testing the UI without model weights.

## Tests

```bash
QT_QPA_PLATFORM=offscreen pytest -q
```

## Project structure

```text
app/                 Qt UI and application controllers
core/capture/        Image, video, camera and RTSP sources
core/inference/      Worker process and IPC manager
core/models/         Unified model adapters and model catalog
core/geometry/       Depth, direct points, filtering and sampling
core/runtime/        State machine, sessions and watchdog
requirements/        Optional model dependency groups
tests/               Unit, UI and integration tests
```
