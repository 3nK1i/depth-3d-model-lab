"""I4.3G2B matched-view MVS-vs-DA3 visual qualification helpers.

This module is deliberately diagnostic-only.  It consumes already frozen
production-native G2B artifacts, verifies that MVS and DA3 share the same
PyCOLMAP-authoritative SfM identity, derives one visual frame from sparse SfM
landmarks only, and prepares blinded fixed-camera A/B review metadata.

It never reruns SfM or either dense provider and never changes authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from core.reconstruction.errors import ReconstructionValidationError

MATCHED_VIEW_SCHEMA_VERSION = 1
MATCHED_VIEW_VIEWS: dict[str, tuple[float, float]] = {
    "oblique": (32.0, 24.0),
    "front": (0.0, 0.0),
    "side": (90.0, 0.0),
    "top": (0.0, 89.0),
}
MATCHED_VIEW_FRAMINGS: dict[str, float] = {
    "detail": 1.45,
    "context": 2.20,
}
MATCHED_VIEW_REVIEW_DIMENSIONS: tuple[str, ...] = (
    "global_completeness",
    "legitimate_hole_recovery",
    "false_geometry_and_floaters",
    "surface_noise_and_roughness",
    "thin_structure_preservation",
    "edge_definition",
    "small_detail_preservation",
    "false_bridging_or_thickening",
)


def sha256_file(path: str | Path) -> str:
    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReconstructionValidationError(f"Could not read {label} {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReconstructionValidationError(f"{label} must contain a JSON object: {source}")
    return payload


def verify_sha256(path: str | Path, expected: str, *, label: str) -> str:
    actual = sha256_file(path)
    if actual != str(expected).lower():
        raise ReconstructionValidationError(
            f"{label} SHA256 mismatch: expected={expected} actual={actual}"
        )
    return actual


@dataclass(frozen=True, slots=True)
class MatchedVisualFrame:
    lower_reconstruction: tuple[float, float, float]
    upper_reconstruction: tuple[float, float, float]
    center_reconstruction: tuple[float, float, float]
    center_viewer: tuple[float, float, float]
    robust_diagonal: float
    lower_quantile: float
    upper_quantile: float

    def camera_distance(self, framing: str) -> float:
        try:
            multiplier = MATCHED_VIEW_FRAMINGS[framing]
        except KeyError as exc:
            raise ReconstructionValidationError(f"Unknown matched-view framing: {framing}") from exc
        return max(float(self.robust_diagonal) * float(multiplier), 0.25)

    def as_dict(self) -> dict[str, Any]:
        return {
            "basis": "authoritative_pycolmap_sparse_landmark_robust_envelope",
            "lower_reconstruction": list(self.lower_reconstruction),
            "upper_reconstruction": list(self.upper_reconstruction),
            "center_reconstruction": list(self.center_reconstruction),
            "center_viewer": list(self.center_viewer),
            "robust_diagonal": self.robust_diagonal,
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile,
            "fov_degrees": 45.0,
            "viewer_transform": "(x, y, z)_reconstruction -> (x, z, -y)_viewer",
            "independent_provider_auto_fit": False,
            "metric_scale_claimed": False,
        }


def derive_shared_visual_frame(
    sfm_snapshot: Mapping[str, Any],
    *,
    lower_quantile: float = 0.02,
    upper_quantile: float = 0.98,
) -> MatchedVisualFrame:
    if not (0.0 <= lower_quantile < upper_quantile <= 1.0):
        raise ReconstructionValidationError("Matched-view sparse quantiles are invalid.")
    raw = sfm_snapshot.get("landmarks")
    if not isinstance(raw, list) or len(raw) < 3:
        raise ReconstructionValidationError("Matched-view requires at least three authoritative SfM landmarks.")
    points: list[np.ndarray] = []
    for item in raw:
        if not isinstance(item, Mapping):
            continue
        value = np.asarray(item.get("position_reconstruction"), dtype=np.float64)
        if value.shape == (3,) and np.all(np.isfinite(value)):
            points.append(value)
    if len(points) < 3:
        raise ReconstructionValidationError("Matched-view authoritative SfM landmarks are invalid.")
    xyz = np.vstack(points)
    lower = np.quantile(xyz, lower_quantile, axis=0)
    upper = np.quantile(xyz, upper_quantile, axis=0)
    extent = upper - lower
    positive = extent[extent > 1e-12]
    if len(positive) < 2:
        raise ReconstructionValidationError("Matched-view authoritative SfM landmark envelope is degenerate.")
    # Give numerically flat axes a tiny display extent without changing center.
    floor = max(float(np.max(positive)) * 1e-3, 1e-6)
    extent = np.maximum(extent, floor)
    center = 0.5 * (lower + upper)
    lower = center - 0.5 * extent
    upper = center + 0.5 * extent
    diagonal = float(np.linalg.norm(upper - lower))
    if not isfinite(diagonal) or diagonal <= 0.0:
        raise ReconstructionValidationError("Matched-view sparse robust diagonal is invalid.")
    center_viewer = np.asarray([center[0], center[2], -center[1]], dtype=np.float64)
    return MatchedVisualFrame(
        lower_reconstruction=tuple(float(v) for v in lower),
        upper_reconstruction=tuple(float(v) for v in upper),
        center_reconstruction=tuple(float(v) for v in center),
        center_viewer=tuple(float(v) for v in center_viewer),
        robust_diagonal=diagonal,
        lower_quantile=float(lower_quantile),
        upper_quantile=float(upper_quantile),
    )


def _artifact_ply(evidence: Mapping[str, Any], *, provider: str) -> tuple[Path, str]:
    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, list):
        raise ReconstructionValidationError(f"{provider} matched-view evidence lacks artifacts.")
    candidates: list[tuple[Path, str]] = []
    for item in artifacts:
        if not isinstance(item, Mapping):
            continue
        path = Path(str(item.get("path", ""))).expanduser().resolve()
        expected = str(item.get("sha256", ""))
        if path.suffix.lower() == ".ply" and expected:
            candidates.append((path, expected))
    if len(candidates) != 1:
        raise ReconstructionValidationError(
            f"{provider} matched-view requires exactly one frozen PLY artifact; found {len(candidates)}."
        )
    path, expected = candidates[0]
    if not path.is_file():
        raise ReconstructionValidationError(f"{provider} matched-view PLY is missing: {path}")
    verify_sha256(path, expected, label=f"{provider} matched-view PLY")
    return path, expected


@dataclass(frozen=True, slots=True)
class MatchedViewSources:
    scenario_id: str
    production_snapshot: Path
    production_snapshot_sha256: str
    production_verdict: Path
    production_verdict_sha256: str
    sfm_snapshot: Path
    sfm_snapshot_sha256: str
    mvs_evidence: Path
    da3_evidence: Path
    mvs_ply: Path
    da3_ply: Path
    mvs_ply_sha256: str
    da3_ply_sha256: str
    frame_id: str
    shared_frame: MatchedVisualFrame


def matched_view_scope_from_frozen_verdict(
    verdict: Mapping[str, Any],
    *,
    scenario_id: str,
) -> str:
    """Validate frozen G2B scenario evidence and return its review scope.

    G2C must not reinterpret an operating-envelope result as a full-scene
    provider comparison.  Complete Room and Low Texture therefore remain
    localized to their authoritative registered-view islands, while
    TempleRing and Weak Overlap use their qualified production-native scope.
    """
    if verdict.get("scenario_id") != scenario_id:
        raise ReconstructionValidationError("Matched-view verdict scenario mismatch.")

    if scenario_id == "g2_temple_ring_mvs_da3":
        if verdict.get("production_native_comparison_valid") is not True:
            raise ReconstructionValidationError("TempleRing matched-view requires valid production-native evidence.")
        if verdict.get("matched_view_diagnostic_authorized") is not True:
            raise ReconstructionValidationError("TempleRing matched-view diagnostic is not authorized.")
        if verdict.get("production_native_rerun_authorized") is not False:
            raise ReconstructionValidationError("TempleRing matched-view requires frozen no-rerun evidence.")
        return "full_authoritative_sfm"

    if scenario_id == "g2_complete_room_mvs_da3":
        required = {
            "scenario_classification": "operating_envelope_failure_authoritative_sfm_coverage",
            "full_room_authoritative_sfm_qualified": False,
            "full_room_provider_comparison_qualified": False,
            "localized_provider_comparison_valid": True,
            "execution_defect_established": False,
            "infrastructure_rerun_authorized": False,
        }
        for key, expected in required.items():
            if verdict.get(key) != expected:
                raise ReconstructionValidationError(
                    f"Complete Room matched-view verdict violation: {key}={verdict.get(key)!r}"
                )
        return "localized_authoritative_registered_views"

    if scenario_id == "g2_low_texture_mvs_da3":
        required = {
            "scenario_classification": "operating_envelope_failure_authoritative_sfm_coverage",
            "authoritative_sfm_qualified_for_full_scenario": False,
            "full_scenario_provider_comparison_qualified": False,
            "localized_provider_comparison_valid": True,
            "execution_defect_established": False,
            "infrastructure_rerun_authorized": False,
        }
        for key, expected in required.items():
            if verdict.get(key) != expected:
                raise ReconstructionValidationError(
                    f"Low Texture matched-view verdict violation: {key}={verdict.get(key)!r}"
                )
        return "localized_authoritative_registered_views"

    if scenario_id == "g2_weak_overlap_mvs_da3":
        required = {
            "authoritative_sfm_connectivity_qualified": True,
            "production_native_comparison_valid": True,
            "mvs_status": "pass",
            "da3_status": "pass",
            "same_authoritative_sfm_shared": True,
            "execution_defect_established": False,
            "infrastructure_rerun_authorized": False,
            "scenario_verdict_pending_g2c_manual_review": True,
        }
        for key, expected in required.items():
            if verdict.get(key) != expected:
                raise ReconstructionValidationError(
                    f"Weak Overlap matched-view verdict violation: {key}={verdict.get(key)!r}"
                )
        return "authoritative_registered_views"

    raise ReconstructionValidationError(f"Unsupported G2C matched-view scenario: {scenario_id}")


def load_matched_view_sources(
    production_snapshot: str | Path,
    production_verdict: str | Path,
    *,
    expected_production_sha256: str,
    expected_verdict_sha256: str,
    expected_scenario_id: str = "g2_temple_ring_mvs_da3",
) -> MatchedViewSources:
    prod_path = Path(production_snapshot).expanduser().resolve()
    verdict_path = Path(production_verdict).expanduser().resolve()
    prod_sha = verify_sha256(prod_path, expected_production_sha256, label="G2B production snapshot")
    verdict_sha = verify_sha256(verdict_path, expected_verdict_sha256, label="G2B production verdict")
    prod = read_json_object(prod_path, label="G2B production snapshot")
    verdict = read_json_object(verdict_path, label="G2B production verdict")
    scenario = prod.get("scenario")
    if not isinstance(scenario, Mapping) or scenario.get("scenario_id") != expected_scenario_id:
        raise ReconstructionValidationError("Matched-view production scenario does not match the requested G2B scenario.")
    if prod.get("status") != "production_native_complete" or prod.get("pass") is not True:
        raise ReconstructionValidationError("Matched-view requires a completed production-native G2B snapshot.")
    matched_view_scope_from_frozen_verdict(
        verdict,
        scenario_id=expected_scenario_id,
    )

    authority = prod.get("authority")
    if not isinstance(authority, Mapping):
        raise ReconstructionValidationError("Matched-view production authority block is missing.")
    required = {
        "camera_pose_authority_owner": "pycolmap_sfm",
        "same_authoritative_sfm_shared_by_both_providers": True,
        "authoritative_sfm_modified": False,
        "provider_selected": False,
        "provider_promoted": False,
        "auto_policy_implemented": False,
        "hybrid_authorized": False,
        "metric_scale_claimed": False,
        "measurement_authoritative": False,
        "certified_metrology": False,
        "surface_mesh_generated": False,
        "permanent_global_view_cap": None,
    }
    for key, expected in required.items():
        if authority.get(key) != expected:
            raise ReconstructionValidationError(f"Matched-view authority violation: {key}={authority.get(key)!r}")

    providers = prod.get("providers")
    if not isinstance(providers, Mapping):
        raise ReconstructionValidationError("Matched-view provider references are missing.")
    mvs_ref = providers.get("colmap_mvs")
    da3_ref = providers.get("da3")
    if not isinstance(mvs_ref, Mapping) or not isinstance(da3_ref, Mapping):
        raise ReconstructionValidationError("Matched-view MVS/DA3 provider blocks are missing.")
    mvs_ev_ref = mvs_ref.get("evidence")
    da3_ev_ref = da3_ref.get("evidence")
    if not isinstance(mvs_ev_ref, Mapping) or not isinstance(da3_ev_ref, Mapping):
        raise ReconstructionValidationError("Matched-view provider evidence references are missing.")
    mvs_ev_path = Path(str(mvs_ev_ref.get("path", ""))).expanduser().resolve()
    da3_ev_path = Path(str(da3_ev_ref.get("path", ""))).expanduser().resolve()
    verify_sha256(mvs_ev_path, str(mvs_ev_ref.get("sha256", "")), label="MVS provider evidence")
    verify_sha256(da3_ev_path, str(da3_ev_ref.get("sha256", "")), label="DA3 provider evidence")
    mvs_ev = read_json_object(mvs_ev_path, label="MVS provider evidence")
    da3_ev = read_json_object(da3_ev_path, label="DA3 provider evidence")
    mvs_sfm = mvs_ev.get("authoritative_sfm_identity")
    da3_sfm = da3_ev.get("authoritative_sfm_identity")
    if not isinstance(mvs_sfm, Mapping) or not isinstance(da3_sfm, Mapping):
        raise ReconstructionValidationError("Matched-view authoritative SfM identities are missing.")
    if mvs_sfm.get("snapshot_sha256") != da3_sfm.get("snapshot_sha256"):
        raise ReconstructionValidationError("Matched-view providers do not share the same authoritative SfM SHA256.")
    if mvs_sfm.get("frame_id") != da3_sfm.get("frame_id"):
        raise ReconstructionValidationError("Matched-view providers do not share the same authoritative SfM frame id.")
    sfm_path = Path(str(mvs_sfm.get("snapshot_path", ""))).expanduser().resolve()
    sfm_sha = str(mvs_sfm.get("snapshot_sha256", ""))
    verify_sha256(sfm_path, sfm_sha, label="authoritative SfM snapshot")
    sfm = read_json_object(sfm_path, label="authoritative SfM snapshot")
    frame = derive_shared_visual_frame(sfm)
    mvs_ply, mvs_ply_sha = _artifact_ply(mvs_ev, provider="MVS")
    da3_ply, da3_ply_sha = _artifact_ply(da3_ev, provider="DA3")
    return MatchedViewSources(
        scenario_id=expected_scenario_id,
        production_snapshot=prod_path,
        production_snapshot_sha256=prod_sha,
        production_verdict=verdict_path,
        production_verdict_sha256=verdict_sha,
        sfm_snapshot=sfm_path,
        sfm_snapshot_sha256=sfm_sha,
        mvs_evidence=mvs_ev_path,
        da3_evidence=da3_ev_path,
        mvs_ply=mvs_ply,
        da3_ply=da3_ply,
        mvs_ply_sha256=mvs_ply_sha,
        da3_ply_sha256=da3_ply_sha,
        frame_id=str(mvs_sfm.get("frame_id")),
        shared_frame=frame,
    )


def blind_assignment(seed_sha256: str) -> dict[str, str]:
    digest = hashlib.sha256((str(seed_sha256) + "|I4.3G2B|matched-view|blind-v1").encode("utf-8")).digest()
    if digest[0] & 1:
        return {"A": "da3", "B": "colmap_mvs"}
    return {"A": "colmap_mvs", "B": "da3"}


def review_template(*, views: Sequence[str], framings: Sequence[str]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "phase": "I4.3G2B",
        "comparison_mode": "matched_view_diagnostic",
        "blinded": True,
        "allowed_values": ["A", "B", "tie", "not_assessable"],
        "review_dimensions": list(MATCHED_VIEW_REVIEW_DIMENSIONS),
        "views": list(views),
        "framings": list(framings),
        "ratings": {dimension: None for dimension in MATCHED_VIEW_REVIEW_DIMENSIONS},
        "notes": "",
        "global_winner_forbidden": True,
        "provider_selection_performed": False,
    }
