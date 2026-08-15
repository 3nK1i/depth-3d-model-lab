from __future__ import annotations

from pathlib import Path

import numpy as np

from app.reconstruction_canonical_cameras import build_canonical_camera_trajectory_snapshot
from core.inspection.contracts import (
    InspectionBoxMethod,
    InspectionBundleSummary,
    InspectionObjectSummary,
    InspectionRunStatus,
)
from core.reconstruction import SemanticGeometryResult, SurfaceMeshResult, persistent_scene_from_pipeline_result
from core.reconstruction.metric_qualification import (
    MetricEvidenceValidationState,
    external_metric_prior_evidence,
)
from core.reconstruction.orchestration import ReconstructionArtifactKind
from core.reconstruction.persistent_authority import build_persistent_scene_authority_snapshot
from tests.phase9_3d_helpers import write_surface_obj
from tests.reconstruction_f5_helpers import run_f5_fixture

SOURCE_SHA256 = "a" * 64
OBSERVATION_ID = "obs-connector_body-connector-main-0"
COORDINATE_FRAME_ID = "camera-depth:sfixture:f1"


def make_phase9_3j_qualified_scene(tmp_path: Path, *, with_pose_uncertainty: bool = True):
    fixture, _providers, result, snapshots = run_f5_fixture(
        tmp_path / "run", request_id="phase9-3j-qualified"
    )
    surface = fixture.store.find_unique(SurfaceMeshResult)
    semantic = fixture.store.find_unique(SemanticGeometryResult)
    surface_path = write_surface_obj(surface, tmp_path / "canonical_surface.obj")
    base = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=snapshots)
    camera_artifact = next(
        item for item in base.artifacts if item.kind is ReconstructionArtifactKind.CAMERA_TRAJECTORY
    )
    matrices = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], 3, axis=0)
    matrices[1, 0, 3] = 0.5
    matrices[2, 0, 3] = 1.0
    updated = dict(snapshots)
    updated["surface_mesh"] = {
        "surface_obj_path": str(surface_path),
        "surface_vertex_count": surface.mesh.vertex_count,
        "surface_face_count": surface.mesh.face_count,
        "metric_geometry_status": "validated_metric",
        "unit_to_metre": 1.0,
        "canonical_geometry_authoritative": True,
        "certified_metrology": False,
    }
    updated["semantic_geometry"] = semantic.as_dict()
    kwargs = {}
    if with_pose_uncertainty:
        kwargs = {
            "translation_uncertainty_m": (0.0005, 0.0005, 0.0005),
            "rotation_uncertainty_rad": (0.002, 0.002, 0.002),
        }
    updated["camera_trajectory"] = build_canonical_camera_trajectory_snapshot(
        matrices_reconstruction=matrices,
        camera_ids=("cam-source", "cam-1", "cam-2"),
        pose_layers=("qualified",) * 3,
        provider_id=camera_artifact.provider_id,
        reconstruction_frame_id="reconstruction-world",
        source_frame_sha256=(SOURCE_SHA256, "b" * 64, "c" * 64),
        **kwargs,
    )
    scene = persistent_scene_from_pipeline_result(result, payload_snapshots_by_kind=updated)
    evidence = external_metric_prior_evidence(
        evidence_id="phase9-3j-metric-prior",
        reconstruction_frame_id="reconstruction-world",
        reference_id="qualified-scale-fixture",
        unit_to_metre=1.0,
        relative_scale_uncertainty=0.01,
        provenance="phase9_3j_test_metric_prior",
        validation_state=MetricEvidenceValidationState.VALIDATED,
        validation_record_id="phase9-3j-scale-validation",
        created_at_utc="2026-08-15T10:00:00+00:00",
    )
    authority = build_persistent_scene_authority_snapshot(
        scene,
        metric_qualification_evidence=evidence,
        created_at_utc="2026-08-15T10:01:00+00:00",
    )
    return scene, authority


def make_phase9_3j_inspection_bundle(tmp_path: Path, *, source_sha256: str = SOURCE_SHA256):
    bundle = (tmp_path / "inspection-bundle").resolve()
    bundle.mkdir(parents=True, exist_ok=True)
    box = {
        "box_id": "box-axis",
        "kind": "axis_aligned",
        "fit_method": "strict_min_max",
        "coordinate_frame_id": COORDINATE_FRAME_ID,
        "centre_xyz_m": [0.10, 0.20, 1.00],
        "extents_xyz_m": [0.042, 0.020, 0.010],
        "orientation_matrix": np.eye(3, dtype=np.float64).tolist(),
        # Real Measure & Inspect bundles persist the eight fitted-box corners
        # consumed by Inspection3DCanvas.  Keep the deterministic 9.3J fixture
        # faithful to that established presentation contract.
        "corners_xyz_m": [
            [0.079, 0.190, 0.995],
            [0.121, 0.190, 0.995],
            [0.079, 0.210, 0.995],
            [0.121, 0.210, 0.995],
            [0.079, 0.190, 1.005],
            [0.121, 0.190, 1.005],
            [0.079, 0.210, 1.005],
            [0.121, 0.210, 1.005],
        ],
    }
    measurements = (
        {
            "kind": "width",
            "status": "available",
            "value_si": 0.042,
            "unit_si": "m",
            "method": "strict_min_max",
            "expanded_uncertainty_si": 0.0015,
            "lower_bound_si": 0.0405,
            "upper_bound_si": 0.0435,
            "reliability": "high",
            "reliability_score": 0.92,
            "valid_observation_count": 5,
            "warnings": (),
            "rejection_reasons": (),
            "uncertainty_basis": "source",
            "uncertainty_qualification": "Expanded source uncertainty",
        },
    )
    obj = InspectionObjectSummary(
        observation_id=OBSERVATION_ID,
        object_index=0,
        class_id=1,
        class_name="connector_body",
        detection_confidence=0.95,
        detection_box_xyxy=(10.0, 10.0, 100.0, 100.0),
        mask_key="mask:0",
        raw_points_key="raw:0",
        filtered_points_key="filtered:0",
        status="success",
        extraction_status="success",
        fitting_status="success",
        measurement_status="available",
        preferred_box_id="box-axis",
        preferred_box_reason="qualified fixture",
        boxes=(("axis_aligned", box),),
        measurements=measurements,
    )
    summary = InspectionBundleSummary(
        bundle_dir=bundle,
        schema_version=1,
        status=InspectionRunStatus.SUCCESS,
        arrays_file="arrays.npz",
        state_file="state.json",
        image_width=640,
        image_height=480,
        depth_model_id="fixture-depth",
        depth_model_name="Fixture Depth",
        perception_model_id="fixture-seg",
        perception_model_name="Fixture Seg",
        depth_scale="metric",
        intrinsics_source="calibrated",
        calibration_status="calibrated",
        confidence_kind="uncertainty",
        box_method=InspectionBoxMethod.AUTO,
        objects=(obj,),
        source_frame_sha256=source_sha256,
    )
    return summary, obj
