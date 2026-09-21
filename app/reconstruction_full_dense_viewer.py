"""I6.7F: independently reverified *complete* admitted PLY -> GPU staging.

Runs off Qt's owner thread. Unlike the provisional I6.7D preview, never
samples vertices or claims persistent-scene, physical-scale or camera authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Mapping

import numpy as np

from app.reconstruction_dense_terminal_verification import verify_dense_completion
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.reconstruction_dense_artifact_preflight import _header

# Hard *allocation* safety budget, not a point subsampling cap: refuse overly
# large artifacts explicitly rather than pretending a downsample is full density.
MAX_STAGING_BYTES = 512 * 1024 * 1024
_READ_BLOCK = 1024 * 1024


@dataclass(frozen=True, slots=True)
class FullDenseViewerFrame:
    session_id: str
    dense_run_id: str
    reconstruction_frame_id: str
    artifact_sha256: str
    points_viewer: np.ndarray
    colors_rgba: np.ndarray
    fit_min: tuple[float, float, float]
    fit_max: tuple[float, float, float]
    preparation_ms: float
    provisional: bool = False
    persistent_scene: bool = False
    metric_scale_claimed: bool = False

    @property
    def point_count(self) -> int:
        return int(self.points_viewer.shape[0])


def _require(ok: bool, detail: str) -> None:
    if not ok:
        raise ReconstructionValidationError('I6.7F full dense viewer: ' + detail)


def prepare_full_dense_viewer_frame(admission: Mapping[str, object], *,
                                    expected_session_id: str) -> FullDenseViewerFrame:
    """Reverify authoritative on-disk admission; load all RGB points off Qt thread.

    The second file digest binds the *same bytes that produced the arrays* to the
    admission; a PLY replaced between E2B1 verification and display is rejected.
    """
    t0 = perf_counter()
    _require(isinstance(admission, Mapping), 'missing verified admission')
    session = str(admission.get('sparse_session_id', ''))
    run_id = str(admission.get('dense_run_id', ''))
    manifest = Path(str(admission.get('manifest_path', '')))
    manifest_sha = str(admission.get('manifest_sha256', ''))
    geometry_sha = str(admission.get('complete_geometry_sha256', ''))
    _require(bool(session) and session == expected_session_id,
             'admission does not belong to the current sparse session')
    _require(bool(run_id) and bool(manifest_sha) and bool(geometry_sha),
             'admission identity or hashes are missing')
    _require(manifest.name == 'final_dense_admission.json' and
             manifest.parent.name == run_id and manifest.parent.parent.name == 'dense_runs',
             'admission location is not a dense run')
    verified = verify_dense_completion(
        offline_output_dir=manifest.parent.parent.parent,
        sparse_session_id=session, dense_run_id=run_id,
        expected_manifest_sha256=manifest_sha, expected_geometry_sha256=geometry_sha,
    )
    _require(verified.vertex_count == admission.get('complete_geometry_vertex_count') and
             verified.artifact_sha256 == geometry_sha and
             verified.manifest_sha256 == manifest_sha,
             'admission changed after independent verification')
    file = Path(verified.geometry_path)
    _require(file.is_file() and not file.is_symlink(), 'admitted PLY is missing or symlinked')
    _require(verified.vertex_count > 0 and verified.vertex_count * 28 <= MAX_STAGING_BYTES,
             'complete geometry exceeds full-resolution memory budget; never silently sample')
    before = file.stat()
    digest = sha256()
    with file.open('rb') as handle:
        fmt, count, dtype, has_rgb = _header(handle, digest)
        _require(fmt in ('binary_little_endian', 'binary_big_endian') and has_rgb,
                 'full RGB viewer requires a binary PLY with RGB')
        _require(count == verified.vertex_count, 'PLY point count changed')
        # Create only two final C-contiguous GPU staging arrays; read vertex rows
        # in bounded chunks. No intermediate full-size float64 or RGB copies.
        positions = np.empty((count, 3), dtype=np.float32)
        colors = np.empty((count, 4), dtype=np.float32)
        chunk = max(1, min(65536, (8 * 1024 * 1024) // dtype.itemsize))
        for start in range(0, count, chunk):
            n = min(chunk, count - start)
            raw = handle.read(n * dtype.itemsize)
            _require(len(raw) == n * dtype.itemsize, 'truncated vertex population')
            digest.update(raw)
            rows = np.frombuffer(raw, dtype=dtype, count=n)
            xyz = positions[start:start+n]
            xyz[:, 0] = rows['x']
            xyz[:, 1] = rows['z']
            xyz[:, 2] = -rows['y']
            rgb = colors[start:start+n]
            rgb[:, 0] = rows['red']
            rgb[:, 1] = rows['green']
            rgb[:, 2] = rows['blue']
            rgb[:, :3] *= np.float32(1.0 / 255.0)
            rgb[:, 3] = 1.0
            _require(np.isfinite(xyz).all() and np.isfinite(rgb).all(),
                     'non-finite GPU buffers')
        while tail := handle.read(_READ_BLOCK):
            digest.update(tail)
    after = file.stat()
    _require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
             (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
             'artifact changed during GPU buffer preparation')
    _require(digest.hexdigest() == verified.artifact_sha256,
             'artifact bytes changed after final admission')
    # Robust fit affects the *camera only*, never omits any points from rendering.
    lower = np.percentile(positions, 1, axis=0)
    upper = np.percentile(positions, 99, axis=0)
    _require(np.isfinite(lower).all() and np.isfinite(upper).all(), 'invalid fit bounds')
    positions.setflags(write=False)
    colors.setflags(write=False)
    return FullDenseViewerFrame(
        session_id=session, dense_run_id=run_id,
        reconstruction_frame_id=verified.reconstruction_frame_id,
        artifact_sha256=verified.artifact_sha256, points_viewer=positions,
        colors_rgba=colors, fit_min=tuple(map(float, lower)),
        fit_max=tuple(map(float, upper)),
        preparation_ms=(perf_counter() - t0) * 1000,
    )
