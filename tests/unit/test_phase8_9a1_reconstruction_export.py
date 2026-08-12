from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.reconstruction import (
    OfflineSparseExportSource,
    ReconstructionArtifactAuthority,
    ReconstructionExportRequest,
    ReconstructionValidationError,
    export_reconstruction_bundle,
    verify_reconstruction_export_bundle,
)


def _offline_payload(tmp_path: Path, *, canonical: bool = False, certified: bool = False) -> dict[str, object]:
    session_id = "5a342ad2-ceeb-40c7-9aeb-ea460bd9160b"
    sparse = (tmp_path / "source" / "sparse_reconstruction.ply").resolve()
    sfm = (tmp_path / "source" / "sfm_result.json").resolve()
    sparse.parent.mkdir(parents=True)
    sparse.write_text(
        "\n".join(
            (
                "ply",
                "format ascii 1.0",
                "comment arbitrary reconstruction gauge; coordinates are NOT metres",
                "element vertex 1",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "0 0 1 180 180 180",
                "",
            )
        ),
        encoding="ascii",
    )
    sfm.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "backend_id": "pycolmap-external",
                "backend_version": "4.1.1",
                "gauge": {
                    "scale_status": "arbitrary",
                    "alignment_status": "arbitrary",
                    "unit_to_metre": None,
                    "scale_source": "",
                    "relative_scale_uncertainty": None,
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "phase": "8.8A3",
        "pass": True,
        "status": "completed",
        "session_id": session_id,
        "camera_pose_available": True,
        "canonical_geometry_authoritative": canonical,
        "persistent_scene_ready": False,
        "certified_metrology": certified,
        "artifacts": {
            "sparse_reconstruction_ply": str(sparse),
            "sfm_result_json": str(sfm),
        },
    }


def test_a1_sparse_export_preserves_supporting_arbitrary_authority(tmp_path: Path) -> None:
    source = OfflineSparseExportSource(_offline_payload(tmp_path))
    result = export_reconstruction_bundle(
        ReconstructionExportRequest(
            output_dir=(tmp_path / "export").resolve(),
            sources=(source,),
        )
    )

    files = verify_reconstruction_export_bundle(result.export_dir)
    assert {item.role for item in files} == {"sparse_point_cloud", "sparse_sfm_result"}
    assert all(item.authority is ReconstructionArtifactAuthority.SUPPORTING_GEOMETRY for item in files)
    assert all(item.canonical_geometry_authoritative is False for item in files)
    assert all(item.certified_metrology is False for item in files)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "eia.reconstruction.export.v1"
    assert manifest["canonical_geometry_authoritative"] is False
    assert manifest["certified_metrology"] is False
    assert manifest["authority_model"]["gaussian_nerf"] == "appearance_only"
    sparse_record = next(item for item in manifest["files"] if item["role"] == "sparse_point_cloud")
    assert sparse_record["metadata"]["scale_status"] == "arbitrary"
    assert sparse_record["metadata"]["unit_to_metre"] is None


def test_a1_sparse_export_rejects_authority_or_metrology_overclaim(tmp_path: Path) -> None:
    with pytest.raises(ReconstructionValidationError, match="canonical geometry"):
        OfflineSparseExportSource(_offline_payload(tmp_path / "canonical", canonical=True))
    with pytest.raises(ReconstructionValidationError, match="certified metrology"):
        OfflineSparseExportSource(_offline_payload(tmp_path / "metrology", certified=True))


def test_a1_sparse_export_requires_arbitrary_gauge(tmp_path: Path) -> None:
    payload = _offline_payload(tmp_path)
    sfm = Path(str(payload["artifacts"]["sfm_result_json"]))  # type: ignore[index]
    raw = json.loads(sfm.read_text(encoding="utf-8"))
    raw["gauge"]["scale_status"] = "metric"
    raw["gauge"]["unit_to_metre"] = 1.0
    sfm.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="arbitrary SfM gauge"):
        OfflineSparseExportSource(payload)


def test_a1_export_fails_closed_on_existing_destination(tmp_path: Path) -> None:
    source = OfflineSparseExportSource(_offline_payload(tmp_path))
    destination = (tmp_path / "existing").resolve()
    destination.mkdir()
    with pytest.raises(ReconstructionValidationError, match="already exists"):
        ReconstructionExportRequest(output_dir=destination, sources=(source,))


def test_a1_export_verification_detects_payload_tampering(tmp_path: Path) -> None:
    source = OfflineSparseExportSource(_offline_payload(tmp_path))
    result = export_reconstruction_bundle(
        ReconstructionExportRequest(
            output_dir=(tmp_path / "export").resolve(),
            sources=(source,),
        )
    )
    sparse = result.export_dir / "geometry" / "sparse_reconstruction.ply"
    sparse.write_bytes(sparse.read_bytes() + b"tampered")
    with pytest.raises(ReconstructionValidationError, match="size changed"):
        verify_reconstruction_export_bundle(result.export_dir)


def test_a1_export_verification_detects_manifest_tampering(tmp_path: Path) -> None:
    source = OfflineSparseExportSource(_offline_payload(tmp_path))
    result = export_reconstruction_bundle(
        ReconstructionExportRequest(
            output_dir=(tmp_path / "export").resolve(),
            sources=(source,),
        )
    )
    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    payload["canonical_geometry_authoritative"] = True
    result.manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReconstructionValidationError, match="manifest checksum mismatch"):
        verify_reconstruction_export_bundle(result.export_dir)
