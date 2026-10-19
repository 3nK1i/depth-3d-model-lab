"""Capability and configuration helpers for the isolated stella_vslam live worker.

Nothing in this module imports or dynamically loads stella_vslam.  The native
library remains outside the production Python process and is consumed only by
the Phase 8.4B worker executable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.contracts.validation import metadata_as_dict
from core.reconstruction.camera import CameraDistortionModel
from core.reconstruction.errors import (
    LiveSlamBackendUnavailableError,
    ReconstructionValidationError,
)
from core.reconstruction.live_contracts import LiveSlamRequest
from core.reconstruction.profiles import ReconstructionQualityProfile

STELLA_LIVE_BACKEND_ID = "stella-vslam-live"
STELLA_LIVE_BACKEND_VERSION = "0.3.13-worker-v2"
STELLA_REQUIRED_VERSION = "0.3.13"
STELLA_QUALIFIED_COMMIT = "7902892d445843563d2d3149505fd3e1a118a141"
STELLA_QUALIFIED_FBOW_COMMIT = "7753c2c17dd7ce8bdf4a7a3fb1262b78d5b9b9b5"
STELLA_PROBE_METADATA_FILE = "DEPTH3D_STELLA_PROBE_INFO.txt"


def _parse_probe_metadata(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            values[key] = value.strip()
    return values


@dataclass(frozen=True, slots=True)
class StellaLiveEnvironmentProbe:
    available: bool
    install_prefix: str
    worker_executable: str
    vocabulary_path: str
    library_path: str
    probe_metadata_path: str
    stella_version: str = ""
    stella_commit: str = ""
    fbow_commit: str = ""
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "install_prefix": self.install_prefix,
            "worker_executable": self.worker_executable,
            "vocabulary_path": self.vocabulary_path,
            "library_path": self.library_path,
            "probe_metadata_path": self.probe_metadata_path,
            "stella_version": self.stella_version,
            "stella_commit": self.stella_commit,
            "fbow_commit": self.fbow_commit,
            "reason": self.reason,
        }


def probe_stella_live_environment(
    install_prefix: str | Path,
    worker_executable: str | Path,
    vocabulary_path: str | Path,
    *,
    require_qualified_build: bool = True,
) -> StellaLiveEnvironmentProbe:
    """Inspect the already-isolated native installation without importing it."""

    prefix = Path(install_prefix).expanduser().resolve()
    worker = Path(worker_executable).expanduser().resolve()
    vocabulary = Path(vocabulary_path).expanduser().resolve()
    library = prefix / "lib" / "libstella_vslam.so"
    metadata_path = prefix / STELLA_PROBE_METADATA_FILE
    metadata = _parse_probe_metadata(metadata_path)

    failures: list[str] = []
    if not prefix.is_dir():
        failures.append("stella_vslam isolated install prefix does not exist")
    if not worker.is_file():
        failures.append("Live SLAM worker executable does not exist")
    elif not worker.stat().st_mode & 0o111:
        failures.append("Live SLAM worker is not executable")
    if not vocabulary.is_file() or vocabulary.stat().st_size <= 0:
        failures.append("FBoW ORB vocabulary is missing or empty")
    if not library.is_file():
        failures.append("isolated libstella_vslam.so is missing")

    version = metadata.get("stella_vslam_version", "")
    commit = metadata.get("stella_vslam_commit", "")
    fbow_commit = metadata.get("stella_FBoW_submodule_commit", "")
    if require_qualified_build:
        if metadata.get("probe_status") != "PASS":
            failures.append("isolated stella_vslam capability probe is not PASS")
        if version != STELLA_REQUIRED_VERSION:
            failures.append(
                f"stella_vslam {STELLA_REQUIRED_VERSION} is required; found {version or 'unknown'}"
            )
        if commit != STELLA_QUALIFIED_COMMIT:
            failures.append("stella_vslam commit does not match the qualified Phase 8.4 build")
        if fbow_commit != STELLA_QUALIFIED_FBOW_COMMIT:
            failures.append("FBoW commit does not match the qualified Phase 8.4 build")

    return StellaLiveEnvironmentProbe(
        available=not failures,
        install_prefix=str(prefix),
        worker_executable=str(worker),
        vocabulary_path=str(vocabulary),
        library_path=str(library),
        probe_metadata_path=str(metadata_path),
        stella_version=version,
        stella_commit=commit,
        fbow_commit=fbow_commit,
        reason="; ".join(failures),
    )


def require_stella_live_environment(
    install_prefix: str | Path,
    worker_executable: str | Path,
    vocabulary_path: str | Path,
    *,
    require_qualified_build: bool = True,
) -> StellaLiveEnvironmentProbe:
    probe = probe_stella_live_environment(
        install_prefix,
        worker_executable,
        vocabulary_path,
        require_qualified_build=require_qualified_build,
    )
    if not probe.available:
        raise LiveSlamBackendUnavailableError(
            f"Isolated stella_vslam live backend is unavailable: {probe.reason}"
        )
    return probe


def _resolved_live_fps(request: LiveSlamRequest) -> float:
    if request.source.expected_frame_rate_hz is not None:
        return float(request.source.expected_frame_rate_hz)
    metadata = metadata_as_dict(request.metadata)  # type: ignore[arg-type]
    candidate = metadata.get("resolved_frame_rate_hz")
    if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
        raise ReconstructionValidationError(
            "stella_vslam live configuration requires an explicit resolved frame rate."
        )
    fps = float(candidate)
    if not (fps > 0.0 and fps < 1000.0):
        raise ReconstructionValidationError(
            "resolved_frame_rate_hz must be finite and between 0 and 1000 Hz."
        )
    return fps


def stella_monocular_config_text(request: LiveSlamRequest) -> str:
    """Translate declared camera metadata to a headless stella YAML config.

    Phase 8.6A may explicitly allow provisional estimated intrinsics for
    experimental tracking. Their provenance remains visible in the generated
    config and they are never relabelled as calibrated.

    Loop detection follows the explicit Phase 8.5 request.  The native worker
    independently verifies the resulting stella API state after startup.  No
    metric scale is introduced here.
    """

    if not isinstance(request, LiveSlamRequest):
        raise TypeError("request must be a LiveSlamRequest.")
    fps = _resolved_live_fps(request)
    camera = request.camera
    matrix = camera.intrinsics.matrix
    fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
    cx, cy = float(matrix[0, 2]), float(matrix[1, 2])
    width, height = camera.image_size.width, camera.image_size.height

    distortion_lines: list[str]
    if camera.distortion_model is CameraDistortionModel.NONE:
        model = "perspective"
        distortion_lines = ["  k1: 0.0", "  k2: 0.0", "  p1: 0.0", "  p2: 0.0", "  k3: 0.0"]
    elif camera.distortion_model is CameraDistortionModel.BROWN_CONRADY:
        coefficients = camera.distortion_coefficients
        if len(coefficients) not in {4, 5}:
            raise ReconstructionValidationError(
                "Phase 8.4B stella_vslam supports Brown-Conrady calibration with 4 or 5 coefficients only; "
                "higher-order OpenCV models must be explicitly converted/calibrated before use."
            )
        k1, k2, p1, p2 = coefficients[:4]
        k3 = coefficients[4] if len(coefficients) == 5 else 0.0
        model = "perspective"
        distortion_lines = [
            f"  k1: {k1:.17g}",
            f"  k2: {k2:.17g}",
            f"  p1: {p1:.17g}",
            f"  p2: {p2:.17g}",
            f"  k3: {k3:.17g}",
        ]
    elif camera.distortion_model is CameraDistortionModel.OPENCV_FISHEYE:
        k1, k2, k3, k4 = camera.distortion_coefficients
        model = "fisheye"
        distortion_lines = [
            f"  k1: {k1:.17g}",
            f"  k2: {k2:.17g}",
            f"  k3: {k3:.17g}",
            f"  k4: {k4:.17g}",
        ]
    else:
        raise ReconstructionValidationError(
            "Backend-specific camera distortion cannot be silently translated to stella_vslam."
        )

    profile_feature_name = {
        ReconstructionQualityProfile.EFFICIENT: "Depth3D live ORB efficient",
        ReconstructionQualityProfile.BALANCED: "Depth3D live ORB balanced",
        ReconstructionQualityProfile.HIGH_QUALITY: "Depth3D live ORB high quality",
    }[request.profile]

    lines = [
        "# Depth & 3D Model Lab Phase 8.6A generated stella_vslam monocular config",
        f"# intrinsics_source={camera.intrinsics.source.value}",
        f"# calibration_status={camera.intrinsics.calibration_status.value}",
        f"# camera_intrinsics_policy={request.camera_intrinsics_policy.value}",
        "Camera:",
        f'  name: "Depth3D {camera.camera_id}"',
        "  setup: monocular",
        f"  model: {model}",
        f"  fx: {fx:.17g}",
        f"  fy: {fy:.17g}",
        f"  cx: {cx:.17g}",
        f"  cy: {cy:.17g}",
        *distortion_lines,
        f"  fps: {fps:.17g}",
        f"  cols: {width}",
        f"  rows: {height}",
        "  color_order: BGR",
        "Feature:",
        f'  name: "{profile_feature_name}"',
        "  scale_factor: 1.2",
        "  num_levels: 8",
        "  ini_fast_threshold: 20",
        "  min_fast_threshold: 7",
        "Mapping:",
        "  backend: g2o",
        "Tracking:",
        "  backend: g2o",
        "LoopDetector:",
        f"  enabled: {'true' if request.enable_loop_closure else 'false'}",
        "  backend: g2o",
        "System:",
        "  map_format: msgpack",
        "",
    ]
    return "\n".join(lines)


def write_stella_monocular_config(request: LiveSlamRequest, path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.write_text(stella_monocular_config_text(request), encoding="utf-8")
    except OSError as exc:
        raise LiveSlamBackendUnavailableError(
            f"Could not write temporary stella_vslam camera configuration: {destination}"
        ) from exc
    return destination
