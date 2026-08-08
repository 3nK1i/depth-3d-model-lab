#!/usr/bin/env python3
"""Phase 8.7C deterministic registration + connector inspection smoke."""

from __future__ import annotations

from dataclasses import replace
import json
from math import cos, radians, sin
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (
    FeatureTolerance,
    InspectionDecision,
    MetricGeometryStatus,
    ReferenceGeometrySource,
    SemanticGeometryRole,
    SfmScaleStatus,
    build_reference_assembly,
    build_semantic_3d_geometry,
    inspect_semantic_geometry,
    write_inspection_report_json,
    write_reference_assembly_json,
)
from scripts.run_semantic_geometry_smoke import (
    _observation_for_region,
    _surface,
)


def _rot_z(degrees_value: float) -> np.ndarray:
    angle = radians(degrees_value)
    c, s = cos(angle), sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rot_x(degrees_value: float) -> np.ndarray:
    angle = radians(degrees_value)
    c, s = cos(angle), sin(angle)
    return np.asarray([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _descriptor(points: np.ndarray):
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    covariance = centered.T @ centered / max(len(points) - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    axes = vectors[:, order]
    for column in range(3):
        pivot = int(np.argmax(np.abs(axes[:, column])))
        if axes[pivot, column] < 0.0:
            axes[:, column] *= -1.0
    if np.linalg.det(axes) < 0.0:
        axes[:, 2] *= -1.0
    epsilon = max(float(values[0]) * 1e-9, 1e-15)
    reliable = bool(
        values[0] > epsilon
        and values[0] / max(values[1], epsilon) >= 1.08
        and values[1] / max(values[2], epsilon) >= 1.25
    )
    return centroid, axes, reliable


def _semantic_fixture():
    surface, camera, scale, translations = _surface(96, 72)
    vertices = surface.mesh.vertices_reconstruction
    body = np.flatnonzero(vertices[:, 0] < -0.25).astype(np.uint32)
    top = np.flatnonzero((vertices[:, 0] > 0.10) & (vertices[:, 1] < -0.16)).astype(np.uint32)
    bottom = np.flatnonzero((vertices[:, 0] > 0.10) & (vertices[:, 1] > 0.16)).astype(np.uint32)
    observations = []
    for frame_index in (0, 2, 4):
        tx = translations[frame_index]
        for class_id, class_name, track_id, indices in (
            (0, "connector_body", "connector-main", body),
            (1, "pin", "pin-top", top),
            (1, "pin", "pin-bottom", bottom),
        ):
            observations.append(
                _observation_for_region(
                    surface,
                    camera,
                    frame_index=frame_index,
                    tx=tx,
                    class_id=class_id,
                    class_name=class_name,
                    track_id=track_id,
                    target_vertex_indices=indices,
                )
            )
    semantic = build_semantic_3d_geometry(
        surface,
        observations,
        role_by_class_name={
            "connector_body": SemanticGeometryRole.CONNECTOR_BODY,
            "pin": SemanticGeometryRole.CONNECTOR_PIN,
        },
    )
    return surface, semantic, scale


def _transform_candidate(
    surface,
    semantic,
    *,
    scale_factor: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    bend_top_degrees: float = 0.0,
    shift_top_reference: np.ndarray | None = None,
    remove_bottom: bool = False,
    low_quality_top: bool = False,
):
    source = surface.mesh.vertices_reconstruction
    vertices = scale_factor * (source @ rotation.T) + translation
    normals = surface.mesh.normals_reconstruction.astype(np.float64) @ rotation.T
    objects = list(semantic.objects)
    top_obj = next((obj for obj in objects if "pin-top" in obj.track_ids), None)
    if bend_top_degrees or shift_top_reference is not None:
        if top_obj is None:
            raise RuntimeError("Requested pin-top deformation but no pin-top object exists.")
        idx = top_obj.vertex_indices.astype(np.int64)
        center = np.mean(vertices[idx], axis=0)
        if bend_top_degrees:
            local_rotation = rotation @ _rot_z(bend_top_degrees) @ rotation.T
            vertices[idx] = (vertices[idx] - center) @ local_rotation.T + center
            normals[idx] = normals[idx] @ local_rotation.T
        if shift_top_reference is not None:
            vertices[idx] += rotation @ (scale_factor * np.asarray(shift_top_reference, dtype=np.float64))

    mesh = replace(
        surface.mesh,
        vertices_reconstruction=vertices,
        normals_reconstruction=normals.astype(np.float32),
    )
    candidate_surface = replace(
        surface,
        reconstruction_frame_id="phase8-7c-candidate",
        mesh=mesh,
    )

    rebuilt = []
    for obj in objects:
        if remove_bottom and "pin-bottom" in obj.track_ids:
            continue
        points = vertices[obj.vertex_indices.astype(np.int64)]
        centroid, axes, reliable = _descriptor(points)
        rebuilt.append(
            replace(
                obj,
                centroid_reconstruction=centroid,
                bounds_min_reconstruction=np.min(points, axis=0),
                bounds_max_reconstruction=np.max(points, axis=0),
                principal_axes_reconstruction=axes,
                orientation_reliable=reliable,
                mean_vertex_quality=(0.20 if low_quality_top and "pin-top" in obj.track_ids else obj.mean_vertex_quality),
            )
        )
    candidate_semantic = replace(
        semantic,
        reconstruction_frame_id="phase8-7c-candidate",
        objects=tuple(rebuilt),
        statistics=replace(semantic.statistics, persistent_object_count=len(rebuilt)),
    )
    return candidate_surface, candidate_semantic


def _metric_variant(surface, semantic, *, profile_id: str):
    unit = 0.01
    rel = 0.020615528128088305
    metric_surface = replace(
        surface,
        scale_profile_id=profile_id,
        metric_geometry_status=MetricGeometryStatus.PROVISIONAL_METRIC,
        scale_status=SfmScaleStatus.ESTIMATED_METRIC,
        unit_to_metre=unit,
        relative_scale_uncertainty=rel,
    )
    metric_objects = tuple(
        replace(
            obj,
            scale_profile_id=profile_id,
            metric_geometry_status=MetricGeometryStatus.PROVISIONAL_METRIC,
            unit_to_metre=unit,
            relative_scale_uncertainty=rel,
        )
        for obj in semantic.objects
    )
    metric_semantic = replace(
        semantic,
        scale_profile_id=profile_id,
        metric_geometry_status=MetricGeometryStatus.PROVISIONAL_METRIC,
        unit_to_metre=unit,
        relative_scale_uncertainty=rel,
        objects=metric_objects,
    )
    return metric_surface, metric_semantic


def main() -> int:
    surface, semantic, scale = _semantic_fixture()
    tolerance = FeatureTolerance(
        position_warn=0.08,
        position_reject=0.16,
        orientation_warn_degrees=5.0,
        orientation_reject_degrees=12.0,
        extent_relative_warn=0.08,
        extent_relative_reject=0.16,
        minimum_geometry_quality=0.60,
        maximum_uncertainty_fraction=0.30,
    )
    reference = build_reference_assembly(
        surface,
        semantic,
        reference_id="phase8-7c-golden-connector",
        source=ReferenceGeometrySource.GOLDEN_RECONSTRUCTION,
        default_tolerance=tolerance,
        registration_anchor_feature_ids=("connector-main",),
    )

    global_scale = 1.30
    global_rotation = _rot_z(17.0)
    global_translation = np.asarray([0.6, -0.4, 0.3], dtype=np.float64)
    pass_surface, pass_semantic = _transform_candidate(
        surface,
        semantic,
        scale_factor=global_scale,
        rotation=global_rotation,
        translation=global_translation,
    )
    pass_report = inspect_semantic_geometry(reference, pass_surface, pass_semantic)

    defect_surface, defect_semantic = _transform_candidate(
        surface,
        semantic,
        scale_factor=global_scale,
        rotation=global_rotation,
        translation=global_translation,
        bend_top_degrees=20.0,
        shift_top_reference=np.asarray([0.0, -0.24, 0.0], dtype=np.float64),
        remove_bottom=True,
    )
    defect_report = inspect_semantic_geometry(reference, defect_surface, defect_semantic)

    weak_surface, weak_semantic = _transform_candidate(
        surface,
        semantic,
        scale_factor=global_scale,
        rotation=global_rotation,
        translation=global_translation,
        low_quality_top=True,
    )
    weak_report = inspect_semantic_geometry(reference, weak_surface, weak_semantic)

    metric_surface, metric_semantic = _metric_variant(
        surface,
        semantic,
        profile_id="phase8-7c-provisional-metric",
    )
    metric_reference = build_reference_assembly(
        metric_surface,
        metric_semantic,
        reference_id="phase8-7c-metric-golden",
        default_tolerance=tolerance,
        registration_anchor_feature_ids=("connector-main",),
    )
    metric_candidate_surface, metric_candidate_semantic = _transform_candidate(
        metric_surface,
        metric_semantic,
        scale_factor=1.0,
        rotation=_rot_z(-9.0),
        translation=np.asarray([-0.2, 0.3, 0.1], dtype=np.float64),
    )
    metric_report = inspect_semantic_geometry(
        metric_reference,
        metric_candidate_surface,
        metric_candidate_semantic,
    )

    defect_by_feature = {value.reference_feature_id: value for value in defect_report.feature_results}
    top = defect_by_feature["pin-top"]
    bottom = defect_by_feature["pin-bottom"]
    weak_top = next(value for value in weak_report.feature_results if value.reference_feature_id == "pin-top")

    output_dir = Path("/tmp/phase8_7c_registration_inspection")
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_json = write_reference_assembly_json(reference, output_dir / "reference.json")
    pass_json = write_inspection_report_json(pass_report, output_dir / "pass_report.json")
    defect_json = write_inspection_report_json(defect_report, output_dir / "defect_report.json")
    weak_json = write_inspection_report_json(weak_report, output_dir / "insufficient_evidence_report.json")

    passed = (
        pass_report.decision is InspectionDecision.PASS
        and pass_report.registration.status.value == "success"
        and abs(pass_report.registration.scale_candidate_to_reference - (1.0 / global_scale)) < 0.03
        and defect_report.decision is InspectionDecision.REJECT
        and "bent_pin" in top.connector_defects
        and "misaligned_pin" in top.connector_defects
        and bottom.decision is InspectionDecision.REJECT
        and "missing_pin" in bottom.connector_defects
        and weak_report.decision is InspectionDecision.INSUFFICIENT_EVIDENCE
        and weak_top.decision is InspectionDecision.INSUFFICIENT_EVIDENCE
        and metric_report.decision is InspectionDecision.PASS
        and metric_report.comparison_metric_available
        and metric_report.reference_unit_to_metre == 0.01
        and metric_report.relative_scale_uncertainty is not None
        and all(
            value.position_error_m is not None
            for value in metric_report.feature_results
            if value.candidate_object_id
        )
        and not pass_report.certified_metrology
        and not metric_report.certified_metrology
        and pass_report.pose_convention == "T_reconstruction_from_camera"
        and reference_json.exists()
        and pass_json.exists()
        and defect_json.exists()
        and weak_json.exists()
        and scale.gauge.unit_to_metre is None
    )

    summary = {
        "phase": "8.7C",
        "pass": passed,
        "relative_pass": pass_report.as_dict(),
        "defect": defect_report.as_dict(),
        "insufficient_evidence": weak_report.as_dict(),
        "metric_pass": metric_report.as_dict(),
        "reference_json": str(reference_json),
        "pass_report_json": str(pass_json),
        "defect_report_json": str(defect_json),
        "insufficient_evidence_report_json": str(weak_json),
        "inspection_qc_enabled": True,
        "reconstruction_ui_enabled": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        "Phase 8.7C registration + inspection/QC smoke: "
        + ("PASS" if passed else "FAIL")
    )
    print(
        f"relative_pass={pass_report.decision.value} "
        f"registration={pass_report.registration.method.value if pass_report.registration.method else 'none'} "
        f"scale={pass_report.registration.scale_candidate_to_reference:.6f}"
    )
    print(
        f"defect_decision={defect_report.decision.value} "
        f"pin_top={','.join(top.connector_defects)} "
        f"pin_bottom={','.join(bottom.connector_defects)}"
    )
    print(
        f"weak_geometry={weak_report.decision.value} "
        f"weak_pin={weak_top.decision.value}"
    )
    print(
        f"metric_decision={metric_report.decision.value} "
        f"unit_to_metre={metric_report.reference_unit_to_metre} "
        f"relative_scale_uncertainty={metric_report.relative_scale_uncertainty:.6f}"
    )
    print("pose_convention=T_reconstruction_from_camera")
    print("certified_metrology=False inspection_qc_enabled=True reconstruction_ui_enabled=False")
    print(f"summary={output_dir / 'summary.json'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
