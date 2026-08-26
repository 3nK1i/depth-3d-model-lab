"""I4.1.6B real dense A/B helpers for localized extra observations.

This module deliberately does not change the production sampling pipeline.
It converts an I4.1.6A-localized source observation plus a real DA3 depth map
into the same arbitrary-gauge SfM frame used by the existing I1 dense pipeline,
then provides compact A/B evidence helpers.  The authoritative geometry remains
the existing PyCOLMAP reconstruction; localized frames are observations only.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np

from core.contracts import ImageSize
from core.reconstruction.dense_fusion import fit_depth_alignment_from_correspondences
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.sfm_contracts import OfflineSfmResult, SfmCameraEstimate, SfmCameraModelEstimate
from core.reconstruction.sfm_coordinates import invert_rigid_matrix
from core.reconstruction.sfm_dense_alignment import (
    SfmDenseDepthAlignmentEvidence,
    SfmDenseDepthAlignmentRejectionCode,
    SfmDenseDepthAlignmentStatus,
)

I4_1_6B_SCHEMA_VERSION = 1


def _bilinear_positive_depth(depth: np.ndarray, *, x: float, y: float) -> float | None:
    height, width = depth.shape
    if x < 0.0 or y < 0.0 or x > width - 1 or y > height - 1:
        return None
    x0 = int(np.floor(x)); y0 = int(np.floor(y))
    x1 = min(x0 + 1, width - 1); y1 = min(y0 + 1, height - 1)
    values = np.asarray(
        [depth[y0, x0], depth[y0, x1], depth[y1, x0], depth[y1, x1]],
        dtype=np.float64,
    )
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        return None
    wx = float(x - x0); wy = float(y - y0)
    top = values[0] * (1.0 - wx) + values[1] * wx
    bottom = values[2] * (1.0 - wx) + values[3] * wx
    sampled = float(top * (1.0 - wy) + bottom * wy)
    return sampled if isfinite(sampled) and sampled > 0.0 else None


def camera_estimate_from_localization(
    localization: Mapping[str, Any],
    *,
    camera_model: SfmCameraModelEstimate,
) -> SfmCameraEstimate:
    """Create an observation-only camera estimate in the authoritative SfM frame."""
    if not isinstance(localization, Mapping):
        raise TypeError("localization must be a mapping.")
    if not isinstance(camera_model, SfmCameraModelEstimate):
        raise TypeError("camera_model must be SfmCameraModelEstimate.")
    if not bool(localization.get("accepted")):
        raise ReconstructionValidationError("I4.1.6B requires accepted I4.1.6A localization evidence.")
    source_index = int(localization.get("source_sequence_index", -1))
    if source_index < 0:
        raise ReconstructionValidationError("I4.1.6B localization source index is invalid.")
    matrix = np.asarray(localization.get("T_reconstruction_from_camera"), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ReconstructionValidationError("I4.1.6B localization pose must be finite 4x4.")
    keyframe_id = str(localization.get("keyframe_id") or f"i4-1-6-localized-{source_index:08d}")
    return SfmCameraEstimate(
        estimate_id=f"i4-1-6b-estimate-{source_index:08d}",
        camera_id=camera_model.camera_id,
        keyframe_id=keyframe_id,
        frame_id=f"i4-1-6b-frame-{source_index:08d}",
        backend_image_id=None,
        T_reconstruction_from_camera=matrix,
    )


def align_localized_candidate_depth(
    sfm_result: OfflineSfmResult,
    *,
    localization: Mapping[str, Any],
    camera_model: SfmCameraModelEstimate,
    dense_depth_sfm_grid: object,
    source_image_width: int,
    source_image_height: int,
    depth_provenance: str,
    minimum_correspondences: int = 8,
    maximum_relative_residual: float = 0.35,
) -> SfmDenseDepthAlignmentEvidence:
    """Fit DA3 depth scale for one PnP-localized frame using its 1.6A inlier landmarks.

    I4.1.6A feature coordinates live on the materialized source frame while the
    DA3 depth is adapted to the shared SfM camera grid. Coordinates are scaled
    explicitly before sampling; no metric scale or second geometry frame is
    introduced.
    """
    if not isinstance(sfm_result, OfflineSfmResult):
        raise TypeError("sfm_result must be OfflineSfmResult.")
    if not isinstance(camera_model, SfmCameraModelEstimate):
        raise TypeError("camera_model must be SfmCameraModelEstimate.")
    minimum = int(minimum_correspondences)
    maximum = float(maximum_relative_residual)
    if minimum < 3:
        raise ReconstructionValidationError("minimum_correspondences must be at least 3.")
    if not isfinite(maximum) or not 0.0 < maximum < 1.0:
        raise ReconstructionValidationError("maximum_relative_residual must be finite in (0,1).")
    width = int(source_image_width); height = int(source_image_height)
    if width <= 0 or height <= 0:
        raise ReconstructionValidationError("source image dimensions must be positive.")
    depth = np.asarray(dense_depth_sfm_grid, dtype=np.float64)
    expected = (camera_model.image_size.height, camera_model.image_size.width)
    if depth.shape != expected or not np.isfinite(depth).all() or np.any(depth <= 0.0):
        raise ReconstructionValidationError("I4.1.6B DA3 depth must be finite positive on the SfM grid.")

    camera = camera_estimate_from_localization(localization, camera_model=camera_model)
    landmark_ids = tuple(str(v) for v in localization.get("matched_landmark_ids", ()))
    feature_xy_raw = tuple(localization.get("matched_feature_xy", ()))
    if len(landmark_ids) != len(feature_xy_raw):
        raise ReconstructionValidationError("I4.1.6B matched landmark/feature evidence is inconsistent.")
    landmark_map = {item.landmark_id: item for item in sfm_result.landmarks}
    T_camera_from_reconstruction = invert_rigid_matrix(camera.T_reconstruction_from_camera)
    sx = float(camera_model.image_size.width) / float(width)
    sy = float(camera_model.image_size.height) / float(height)

    source_depths: list[float] = []
    camera_depths: list[float] = []
    candidate = 0; behind = 0; out_or_missing = 0; invalid = 0
    for landmark_id, xy in zip(landmark_ids, feature_xy_raw):
        candidate += 1
        landmark = landmark_map.get(landmark_id)
        if landmark is None:
            out_or_missing += 1
            continue
        position_camera = (
            T_camera_from_reconstruction[:3, :3] @ landmark.position_reconstruction
            + T_camera_from_reconstruction[:3, 3]
        )
        z = float(position_camera[2])
        if not isfinite(z) or z <= 0.0:
            behind += 1
            continue
        try:
            x = float(xy[0]) * sx; y = float(xy[1]) * sy
        except (TypeError, ValueError, IndexError) as exc:
            raise ReconstructionValidationError("I4.1.6B feature coordinates must be numeric pairs.") from exc
        sampled = _bilinear_positive_depth(depth, x=x, y=y)
        if sampled is None:
            invalid += 1
            continue
        source_depths.append(float(sampled)); camera_depths.append(z)

    valid = len(source_depths)
    common = dict(
        session_id=sfm_result.session_id,
        keyframe_id=camera.keyframe_id,
        frame_id=camera.frame_id,
        camera_id=camera.camera_id,
        camera_model_name=camera_model.backend_model_name,
        depth_provenance=str(depth_provenance),
        sfm_scale_status=sfm_result.gauge.scale_status,
        sfm_world_alignment_status=sfm_result.gauge.alignment_status,
        depth_image_size=ImageSize(width=expected[1], height=expected[0]),
        minimum_correspondences=minimum,
        maximum_relative_residual=maximum,
        candidate_observations=candidate,
        behind_camera_observations=behind,
        out_of_bounds_observations=out_or_missing,
        invalid_depth_observations=invalid,
        valid_correspondences=valid,
    )
    if valid < minimum:
        return SfmDenseDepthAlignmentEvidence(
            **common,
            status=SfmDenseDepthAlignmentStatus.REJECTED,
            inlier_correspondences=0,
            alignment=None,
            rejection_code=SfmDenseDepthAlignmentRejectionCode.INSUFFICIENT_CORRESPONDENCES,
            message=f"Only {valid} localized SfM/depth correspondences remain; {minimum} required.",
        )
    provenance = (
        f"{depth_provenance};i4_1_6b_localized_observation;"
        f"sfm_backend={sfm_result.backend_id};sfm_keyframe={camera.keyframe_id};"
        f"reconstruction_gauge={sfm_result.gauge.scale_status.value}"
    )
    try:
        fit = fit_depth_alignment_from_correspondences(
            source_depths,
            camera_depths,
            provenance=provenance,
            minimum_correspondences=minimum,
            maximum_relative_residual=maximum,
        )
    except ReconstructionValidationError as exc:
        return SfmDenseDepthAlignmentEvidence(
            **common,
            status=SfmDenseDepthAlignmentStatus.REJECTED,
            inlier_correspondences=0,
            alignment=None,
            rejection_code=SfmDenseDepthAlignmentRejectionCode.ROBUST_SCALE_FIT_REJECTED,
            message=f"Robust localized sparse/depth scale fit rejected the frame: {exc}",
        )
    return SfmDenseDepthAlignmentEvidence(
        **common,
        status=SfmDenseDepthAlignmentStatus.ACCEPTED,
        inlier_correspondences=fit.inlier_correspondences,
        robust_relative_residual_limit=fit.robust_relative_residual_limit,
        median_relative_residual=fit.median_relative_residual,
        p90_relative_residual=fit.p90_relative_residual,
        alignment=fit.alignment,
        rejection_code=None,
        message=(
            "Localized DA3 depth aligned to the existing arbitrary-gauge SfM frame using "
            "I4.1.6A PnP-inlier landmarks; no metric authority promoted."
        ),
    )


@dataclass(frozen=True, slots=True)
class DenseABPlanSummary:
    plan_name: str
    planned_candidates: int
    alignment_accepted: int
    pair_edges: int
    pair_pass: int
    pair_degraded: int
    pair_inconclusive: int
    selected_qualified_views: int
    full_span: bool
    final_fusion_cap: int = 40

    def __post_init__(self) -> None:
        counts = (
            self.planned_candidates, self.alignment_accepted, self.pair_edges,
            self.pair_pass, self.pair_degraded, self.pair_inconclusive,
            self.selected_qualified_views, self.final_fusion_cap,
        )
        if any(isinstance(v, bool) or int(v) < 0 for v in counts):
            raise ReconstructionValidationError("I4.1.6B summary counts must be non-negative integers.")
        if self.alignment_accepted > self.planned_candidates:
            raise ReconstructionValidationError("alignment_accepted cannot exceed planned_candidates.")
        if self.pair_pass + self.pair_degraded + self.pair_inconclusive != self.pair_edges:
            raise ReconstructionValidationError("I4.1.6B pair accounting must be exact.")
        if self.selected_qualified_views > self.alignment_accepted:
            raise ReconstructionValidationError("selected views cannot exceed aligned views.")
        if self.final_fusion_cap < 1:
            raise ReconstructionValidationError("final_fusion_cap must be positive.")

    @property
    def alignment_fraction(self) -> float:
        return float(self.alignment_accepted / max(1, self.planned_candidates))

    @property
    def pair_pass_fraction(self) -> float | None:
        decisive = self.pair_pass + self.pair_degraded
        return None if decisive == 0 else float(self.pair_pass / decisive)

    @property
    def dense_gate_pass(self) -> bool:
        return bool(
            self.full_span
            and self.alignment_fraction >= 0.80
            and self.selected_qualified_views >= 10
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "plan_name": self.plan_name,
            "planned_candidates": int(self.planned_candidates),
            "alignment_accepted": int(self.alignment_accepted),
            "alignment_fraction": self.alignment_fraction,
            "pair_edges": int(self.pair_edges),
            "pair_pass": int(self.pair_pass),
            "pair_degraded": int(self.pair_degraded),
            "pair_inconclusive": int(self.pair_inconclusive),
            "pair_pass_fraction": self.pair_pass_fraction,
            "selected_qualified_views": int(self.selected_qualified_views),
            "full_span": bool(self.full_span),
            "final_fusion_cap": int(self.final_fusion_cap),
            "fusion_eligible_view_count": int(min(self.selected_qualified_views, self.final_fusion_cap)),
            "dense_gate_pass": self.dense_gate_pass,
        }


_PAIR_FRACTION_EPS = 1e-12


def compare_dense_plan_to_baseline(
    *,
    selected_views: int,
    pair_pass_fraction: float,
    baseline_selected_views: int,
    baseline_pair_pass_fraction: float,
) -> dict[str, Any]:
    """Compare dense-plan evidence without making a saturated baseline impossible to beat.

    Ordinary baselines still require a strict pair-PASS improvement.  When the
    global-only baseline is already mathematically saturated at 1.0, a local
    plan can instead satisfy the pair-quality requirement by preserving the
    saturated quality exactly (within floating-point tolerance) while adding
    strictly more qualified views.  This is a comparison-rule correction only;
    no dense-consistency threshold is relaxed.
    """

    if isinstance(selected_views, bool) or int(selected_views) < 0:
        raise ReconstructionValidationError("I4.1.6B selected-view evidence is invalid.")
    if isinstance(baseline_selected_views, bool) or int(baseline_selected_views) < 0:
        raise ReconstructionValidationError("I4.1.6B baseline selected-view evidence is invalid.")
    pair_fraction = float(pair_pass_fraction)
    baseline_pair = float(baseline_pair_pass_fraction)
    if not isfinite(pair_fraction) or not 0.0 <= pair_fraction <= 1.0:
        raise ReconstructionValidationError("I4.1.6B pair-PASS evidence is invalid.")
    if not isfinite(baseline_pair) or not 0.0 <= baseline_pair <= 1.0:
        raise ReconstructionValidationError("I4.1.6B baseline pair-PASS evidence is invalid.")

    selected_gain = int(selected_views) - int(baseline_selected_views)
    pair_gain = float(pair_fraction - baseline_pair)
    baseline_saturated = bool(baseline_pair >= 1.0 - _PAIR_FRACTION_EPS)
    pair_quality_regressed = bool(pair_fraction + _PAIR_FRACTION_EPS < baseline_pair)
    strict_pair_improvement = bool(pair_gain > _PAIR_FRACTION_EPS)
    saturated_no_regression = bool(baseline_saturated and not pair_quality_regressed)
    pair_quality_requirement_pass = bool(strict_pair_improvement or saturated_no_regression)
    comparison_pass = bool(selected_gain > 0 and pair_quality_requirement_pass)
    comparison_rule = (
        "saturated_baseline_no_pair_regression_plus_selected_gain"
        if baseline_saturated
        else "strict_pair_pass_fraction_gain_plus_selected_gain"
    )
    return {
        "comparison_pass": comparison_pass,
        "comparison_rule": comparison_rule,
        "selected_view_gain_over_global_only": selected_gain,
        "pair_pass_fraction_gain_over_global_only": pair_gain,
        "baseline_pair_pass_fraction_saturated": baseline_saturated,
        "pair_quality_regressed": pair_quality_regressed,
        "pair_quality_requirement_pass": pair_quality_requirement_pass,
    }


def choose_real_dense_plan(
    plans: Sequence[DenseABPlanSummary],
    *,
    baseline_selected_views: int,
    baseline_pair_pass_fraction: float,
) -> dict[str, Any]:
    """Choose an A/B winner without changing production authority."""
    candidates = tuple(plans)
    if not candidates:
        raise ReconstructionValidationError("I4.1.6B requires at least one real local plan.")
    if baseline_selected_views < 0 or not 0.0 <= float(baseline_pair_pass_fraction) <= 1.0:
        raise ReconstructionValidationError("I4.1.6B baseline evidence is invalid.")

    def key(item: DenseABPlanSummary) -> tuple[int, int, float, float, int]:
        pair_fraction = item.pair_pass_fraction or 0.0
        return (
            1 if item.dense_gate_pass else 0,
            int(item.selected_qualified_views),
            float(pair_fraction),
            float(item.alignment_fraction),
            -int(item.planned_candidates),
        )

    winner = max(candidates, key=key)
    pair_fraction = winner.pair_pass_fraction or 0.0
    comparison = compare_dense_plan_to_baseline(
        selected_views=int(winner.selected_qualified_views),
        pair_pass_fraction=float(pair_fraction),
        baseline_selected_views=int(baseline_selected_views),
        baseline_pair_pass_fraction=float(baseline_pair_pass_fraction),
    )
    validated = bool(winner.dense_gate_pass and comparison["comparison_pass"])
    return {
        "recommended_plan": winner.plan_name,
        "real_dense_ab_validation_pass": validated,
        **comparison,
        "i4_1_6c_production_binding_authorized": validated,
        "production_sampling_integration_authorized": False,
        "dense_consistency_threshold_relaxation_authorized": False,
        "i4_2_detail_preservation_authorized": False,
        "mesh_authorization": False,
    }
