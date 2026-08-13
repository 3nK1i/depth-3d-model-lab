"""Phase 9.1H persistent calibration/metric authority sidecar.

The mature Phase 8 persistent reconstruction scene manifest remains unchanged.
Phase 9.1H stores application-level camera-calibration and metric-qualification
evidence in a separate, integrity-checked sidecar that is cryptographically
bound to the persistent scene identity, fingerprint, and session.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from core.reconstruction.calibration_profile import (
    CameraCalibrationProfile,
    CameraCalibrationValidationState,
    camera_calibration_profile_from_dict,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.metric_qualification import (
    MetricQualificationAssessment,
    MetricQualificationEvidence,
    assess_metric_qualification,
    metric_qualification_evidence_from_dict,
)
from core.reconstruction.persistent_scene import PersistentReconstructionScene
from core.reconstruction.validation import require_iso8601_utc, require_sha256, require_uuid
from core.contracts.validation import require_nonempty


PERSISTENT_SCENE_AUTHORITY_SCHEMA_VERSION = 1
PERSISTENT_SCENE_AUTHORITY_PHASE = "9.1H"
PERSISTENT_SCENE_AUTHORITY_MANIFEST = "persistent_reconstruction_authority.json"
PERSISTENT_SCENE_AUTHORITY_CHECKSUM = "persistent_reconstruction_authority.json.sha256"


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_scale_profile_id(evidence_id: str) -> str:
    digest = hashlib.sha256(str(evidence_id).encode("utf-8")).hexdigest()[:20]
    return f"phase9-persisted-scale-{digest}"


def _authority_payload(
    *,
    scene_id: str,
    scene_fingerprint_sha256: str,
    session_id: str,
    camera_calibration_profile: CameraCalibrationProfile | None,
    metric_qualification_evidence: MetricQualificationEvidence | None,
    metric_assessment_scale_profile_id: str,
    metric_qualification_assessment: MetricQualificationAssessment | None,
) -> dict[str, Any]:
    return {
        "schema_version": PERSISTENT_SCENE_AUTHORITY_SCHEMA_VERSION,
        "phase": PERSISTENT_SCENE_AUTHORITY_PHASE,
        "scene_id": scene_id,
        "scene_fingerprint_sha256": scene_fingerprint_sha256,
        "session_id": session_id,
        "camera_calibration_profile": (
            None if camera_calibration_profile is None else camera_calibration_profile.as_dict()
        ),
        "metric_qualification_evidence": (
            None if metric_qualification_evidence is None else metric_qualification_evidence.as_dict()
        ),
        "metric_assessment_scale_profile_id": metric_assessment_scale_profile_id,
        "metric_qualification_assessment": (
            None if metric_qualification_assessment is None else metric_qualification_assessment.as_dict()
        ),
        "certified_metrology": False,
    }


@dataclass(frozen=True, slots=True)
class PersistentSceneAuthoritySnapshot:
    """Immutable Phase 9 authority evidence bound to one persistent scene."""

    authority_id: str
    scene_id: str
    scene_fingerprint_sha256: str
    session_id: str
    camera_calibration_profile: CameraCalibrationProfile | None
    metric_qualification_evidence: MetricQualificationEvidence | None
    metric_assessment_scale_profile_id: str
    metric_qualification_assessment: MetricQualificationAssessment | None
    authority_fingerprint_sha256: str
    created_at_utc: str
    certified_metrology: bool = False
    schema_version: int = PERSISTENT_SCENE_AUTHORITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        scene_id = require_nonempty(self.scene_id, field_name="scene_id")
        scene_fingerprint = require_sha256(
            self.scene_fingerprint_sha256,
            field_name="scene_fingerprint_sha256",
            allow_empty=False,
        )
        session_id = require_uuid(self.session_id, field_name="session_id")
        if self.camera_calibration_profile is not None and not isinstance(
            self.camera_calibration_profile, CameraCalibrationProfile
        ):
            raise TypeError("camera_calibration_profile must be CameraCalibrationProfile or None.")
        if (
            self.camera_calibration_profile is not None
            and self.camera_calibration_profile.validation_state
            is CameraCalibrationValidationState.REJECTED
        ):
            raise ReconstructionValidationError(
                "Rejected camera-calibration evidence cannot be persisted as the active scene profile."
            )
        evidence = self.metric_qualification_evidence
        assessment = self.metric_qualification_assessment
        scale_profile_id = str(self.metric_assessment_scale_profile_id).strip()
        if evidence is None:
            if assessment is not None or scale_profile_id:
                raise ReconstructionValidationError(
                    "Metric assessment/scale profile ID cannot exist without metric evidence."
                )
        else:
            if not isinstance(evidence, MetricQualificationEvidence):
                raise TypeError("metric_qualification_evidence must be MetricQualificationEvidence or None.")
            if not scale_profile_id:
                raise ReconstructionValidationError(
                    "Metric evidence persistence requires metric_assessment_scale_profile_id."
                )
            expected = assess_metric_qualification(evidence, scale_profile_id=scale_profile_id)
            if assessment is None:
                raise ReconstructionValidationError(
                    "Metric evidence persistence requires its derived qualification assessment."
                )
            if not isinstance(assessment, MetricQualificationAssessment):
                raise TypeError("metric_qualification_assessment must be MetricQualificationAssessment or None.")
            if assessment.as_dict() != expected.as_dict():
                raise ReconstructionValidationError(
                    "Persisted metric qualification assessment does not match the evidence."
                )
        if bool(self.certified_metrology):
            raise ReconstructionValidationError(
                "Phase 9.1H persistence cannot claim certified metrology."
            )
        if int(self.schema_version) != PERSISTENT_SCENE_AUTHORITY_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported persistent authority schema version.")
        created = require_iso8601_utc(self.created_at_utc, field_name="created_at_utc")
        fingerprint = require_sha256(
            self.authority_fingerprint_sha256,
            field_name="authority_fingerprint_sha256",
            allow_empty=False,
        )
        expected_fingerprint = hashlib.sha256(
            _canonical_json(
                _authority_payload(
                    scene_id=scene_id,
                    scene_fingerprint_sha256=scene_fingerprint,
                    session_id=session_id,
                    camera_calibration_profile=self.camera_calibration_profile,
                    metric_qualification_evidence=evidence,
                    metric_assessment_scale_profile_id=scale_profile_id,
                    metric_qualification_assessment=assessment,
                )
            )
        ).hexdigest()
        if fingerprint != expected_fingerprint:
            raise ReconstructionValidationError(
                "Persistent authority fingerprint does not match normalized authority evidence."
            )
        expected_id = f"pra-{fingerprint[:24]}"
        if self.authority_id != expected_id:
            raise ReconstructionValidationError(
                "authority_id must be derived from authority_fingerprint_sha256."
            )
        object.__setattr__(self, "scene_id", scene_id)
        object.__setattr__(self, "scene_fingerprint_sha256", scene_fingerprint)
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "metric_assessment_scale_profile_id", scale_profile_id)
        object.__setattr__(self, "authority_fingerprint_sha256", fingerprint)
        object.__setattr__(self, "created_at_utc", created)
        object.__setattr__(self, "certified_metrology", False)
        object.__setattr__(self, "schema_version", PERSISTENT_SCENE_AUTHORITY_SCHEMA_VERSION)

    @property
    def metric_coordinates_allowed(self) -> bool:
        assessment = self.metric_qualification_assessment
        return bool(assessment and assessment.authorizes_metric_coordinates)

    @property
    def calibrated_metric_scale(self) -> bool:
        assessment = self.metric_qualification_assessment
        return bool(assessment and assessment.authorizes_calibrated_metric_scale)

    def as_dict(self) -> dict[str, Any]:
        payload = _authority_payload(
            scene_id=self.scene_id,
            scene_fingerprint_sha256=self.scene_fingerprint_sha256,
            session_id=self.session_id,
            camera_calibration_profile=self.camera_calibration_profile,
            metric_qualification_evidence=self.metric_qualification_evidence,
            metric_assessment_scale_profile_id=self.metric_assessment_scale_profile_id,
            metric_qualification_assessment=self.metric_qualification_assessment,
        )
        payload.update(
            {
                "authority_id": self.authority_id,
                "authority_fingerprint_sha256": self.authority_fingerprint_sha256,
                "created_at_utc": self.created_at_utc,
            }
        )
        return payload


def build_persistent_scene_authority_snapshot(
    scene: PersistentReconstructionScene,
    *,
    camera_calibration_profile: CameraCalibrationProfile | None = None,
    metric_qualification_evidence: MetricQualificationEvidence | None = None,
    metric_assessment_scale_profile_id: str | None = None,
    created_at_utc: str | None = None,
) -> PersistentSceneAuthoritySnapshot:
    if not isinstance(scene, PersistentReconstructionScene):
        raise TypeError("scene must be PersistentReconstructionScene.")
    if camera_calibration_profile is not None and not isinstance(
        camera_calibration_profile, CameraCalibrationProfile
    ):
        raise TypeError("camera_calibration_profile must be CameraCalibrationProfile or None.")
    if metric_qualification_evidence is not None and not isinstance(
        metric_qualification_evidence, MetricQualificationEvidence
    ):
        raise TypeError("metric_qualification_evidence must be MetricQualificationEvidence or None.")
    scale_profile_id = ""
    assessment = None
    if metric_qualification_evidence is not None:
        scale_profile_id = (
            str(metric_assessment_scale_profile_id).strip()
            if metric_assessment_scale_profile_id is not None
            else _default_scale_profile_id(metric_qualification_evidence.evidence_id)
        )
        if not scale_profile_id:
            raise ReconstructionValidationError("metric_assessment_scale_profile_id cannot be empty.")
        assessment = assess_metric_qualification(
            metric_qualification_evidence,
            scale_profile_id=scale_profile_id,
        )
    fingerprint_payload = _authority_payload(
        scene_id=scene.scene_id,
        scene_fingerprint_sha256=scene.scene_fingerprint_sha256,
        session_id=scene.session_id,
        camera_calibration_profile=camera_calibration_profile,
        metric_qualification_evidence=metric_qualification_evidence,
        metric_assessment_scale_profile_id=scale_profile_id,
        metric_qualification_assessment=assessment,
    )
    fingerprint = hashlib.sha256(_canonical_json(fingerprint_payload)).hexdigest()
    return PersistentSceneAuthoritySnapshot(
        authority_id=f"pra-{fingerprint[:24]}",
        scene_id=scene.scene_id,
        scene_fingerprint_sha256=scene.scene_fingerprint_sha256,
        session_id=scene.session_id,
        camera_calibration_profile=camera_calibration_profile,
        metric_qualification_evidence=metric_qualification_evidence,
        metric_assessment_scale_profile_id=scale_profile_id,
        metric_qualification_assessment=assessment,
        authority_fingerprint_sha256=fingerprint,
        created_at_utc=created_at_utc or datetime.now(timezone.utc).isoformat(),
    )


def persistent_scene_authority_snapshot_from_dict(
    raw: Mapping[str, Any],
) -> PersistentSceneAuthoritySnapshot:
    try:
        if int(raw["schema_version"]) != PERSISTENT_SCENE_AUTHORITY_SCHEMA_VERSION:
            raise ReconstructionValidationError("Unsupported persistent authority schema version.")
        calibration_raw = raw.get("camera_calibration_profile")
        evidence_raw = raw.get("metric_qualification_evidence")
        assessment_raw = raw.get("metric_qualification_assessment")
        calibration = (
            None
            if calibration_raw is None
            else camera_calibration_profile_from_dict(_mapping(calibration_raw, "camera_calibration_profile"))
        )
        evidence = (
            None
            if evidence_raw is None
            else metric_qualification_evidence_from_dict(_mapping(evidence_raw, "metric_qualification_evidence"))
        )
        scale_profile_id = str(raw.get("metric_assessment_scale_profile_id", "")).strip()
        assessment = None
        if evidence is not None:
            assessment = assess_metric_qualification(evidence, scale_profile_id=scale_profile_id)
            if assessment_raw is None:
                raise ReconstructionValidationError(
                    "Persistent metric evidence is missing its assessment snapshot."
                )
            if _mapping(assessment_raw, "metric_qualification_assessment") != assessment.as_dict():
                raise ReconstructionValidationError(
                    "Persistent metric assessment snapshot does not match recomputed evidence authority."
                )
        elif assessment_raw is not None:
            raise ReconstructionValidationError(
                "Persistent metric assessment cannot exist without metric evidence."
            )
        return PersistentSceneAuthoritySnapshot(
            authority_id=str(raw["authority_id"]),
            scene_id=str(raw["scene_id"]),
            scene_fingerprint_sha256=str(raw["scene_fingerprint_sha256"]),
            session_id=str(raw["session_id"]),
            camera_calibration_profile=calibration,
            metric_qualification_evidence=evidence,
            metric_assessment_scale_profile_id=scale_profile_id,
            metric_qualification_assessment=assessment,
            authority_fingerprint_sha256=str(raw["authority_fingerprint_sha256"]),
            created_at_utc=str(raw["created_at_utc"]),
            certified_metrology=bool(raw.get("certified_metrology", False)),
            schema_version=int(raw["schema_version"]),
        )
    except KeyError as exc:
        raise ReconstructionValidationError(
            f"Persistent authority snapshot is missing required field: {exc.args[0]}."
        ) from exc
    except ReconstructionValidationError:
        raise
    except (TypeError, ValueError) as exc:
        raise ReconstructionValidationError(
            f"Persistent authority snapshot contains an invalid value: {exc}."
        ) from exc


def write_persistent_scene_authority_snapshot(
    snapshot: PersistentSceneAuthoritySnapshot,
    output_dir: str | Path,
) -> Path:
    if not isinstance(snapshot, PersistentSceneAuthoritySnapshot):
        raise TypeError("snapshot must be PersistentSceneAuthoritySnapshot.")
    root = Path(output_dir).expanduser()
    if not root.is_absolute():
        raise ReconstructionValidationError("Persistent authority output_dir must be absolute.")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / PERSISTENT_SCENE_AUTHORITY_MANIFEST
    checksum = root / PERSISTENT_SCENE_AUTHORITY_CHECKSUM
    payload = json.dumps(snapshot.as_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    manifest_tmp = root / f".{PERSISTENT_SCENE_AUTHORITY_MANIFEST}.{os.getpid()}.tmp"
    checksum_tmp = root / f".{PERSISTENT_SCENE_AUTHORITY_CHECKSUM}.{os.getpid()}.tmp"
    try:
        manifest_tmp.write_text(payload, encoding="utf-8")
        manifest_sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        checksum_tmp.write_text(
            f"{manifest_sha}  {PERSISTENT_SCENE_AUTHORITY_MANIFEST}\n",
            encoding="utf-8",
        )
        os.replace(manifest_tmp, manifest)
        os.replace(checksum_tmp, checksum)
    finally:
        manifest_tmp.unlink(missing_ok=True)
        checksum_tmp.unlink(missing_ok=True)
    return manifest


def read_persistent_scene_authority_snapshot(
    path: str | Path,
    *,
    expected_scene: PersistentReconstructionScene | None = None,
) -> PersistentSceneAuthoritySnapshot:
    target = Path(path).expanduser()
    if target.is_dir():
        target = target / PERSISTENT_SCENE_AUTHORITY_MANIFEST
    if not target.is_absolute():
        raise ReconstructionValidationError("Persistent authority path must be absolute.")
    target = target.resolve()
    if not target.is_file():
        raise ReconstructionValidationError(f"Persistent authority manifest does not exist: {target}")
    checksum = target.parent / PERSISTENT_SCENE_AUTHORITY_CHECKSUM
    if not checksum.is_file():
        raise ReconstructionValidationError("Persistent authority checksum sidecar is missing.")
    parts = checksum.read_text(encoding="utf-8").strip().split()
    if len(parts) < 2 or parts[1] != PERSISTENT_SCENE_AUTHORITY_MANIFEST:
        raise ReconstructionValidationError("Persistent authority checksum sidecar is malformed.")
    expected_sha = require_sha256(parts[0], field_name="authority_manifest_sha256", allow_empty=False)
    if _sha256_file(target) != expected_sha:
        raise ReconstructionValidationError("Persistent authority manifest checksum mismatch.")
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ReconstructionValidationError("Persistent authority manifest is not valid JSON.") from exc
    if not isinstance(raw, Mapping):
        raise ReconstructionValidationError("Persistent authority manifest root must be an object.")
    snapshot = persistent_scene_authority_snapshot_from_dict(raw)
    if expected_scene is not None:
        if not isinstance(expected_scene, PersistentReconstructionScene):
            raise TypeError("expected_scene must be PersistentReconstructionScene or None.")
        if snapshot.scene_id != expected_scene.scene_id:
            raise ReconstructionValidationError("Persistent authority scene_id does not match the scene manifest.")
        if snapshot.scene_fingerprint_sha256 != expected_scene.scene_fingerprint_sha256:
            raise ReconstructionValidationError(
                "Persistent authority scene fingerprint does not match the scene manifest."
            )
        if snapshot.session_id != expected_scene.session_id:
            raise ReconstructionValidationError("Persistent authority session ID does not match the scene manifest.")
    return snapshot


def read_optional_persistent_scene_authority_snapshot(
    scene_manifest: str | Path,
    *,
    expected_scene: PersistentReconstructionScene,
) -> tuple[PersistentSceneAuthoritySnapshot | None, Path | None]:
    scene_path = Path(scene_manifest).expanduser().resolve()
    root = scene_path if scene_path.is_dir() else scene_path.parent
    manifest = root / PERSISTENT_SCENE_AUTHORITY_MANIFEST
    checksum = root / PERSISTENT_SCENE_AUTHORITY_CHECKSUM
    if not manifest.exists() and not checksum.exists():
        return None, None
    if manifest.exists() != checksum.exists():
        raise ReconstructionValidationError(
            "Persistent authority sidecar is incomplete; manifest/checksum must either both exist or both be absent."
        )
    return (
        read_persistent_scene_authority_snapshot(manifest, expected_scene=expected_scene),
        manifest.resolve(),
    )


def remove_persistent_scene_authority_snapshot(output_dir: str | Path) -> None:
    root = Path(output_dir).expanduser()
    if not root.is_absolute():
        raise ReconstructionValidationError("Persistent authority output_dir must be absolute.")
    root = root.resolve()
    (root / PERSISTENT_SCENE_AUTHORITY_MANIFEST).unlink(missing_ok=True)
    (root / PERSISTENT_SCENE_AUTHORITY_CHECKSUM).unlink(missing_ok=True)


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReconstructionValidationError(f"{field_name} must be an object.")
    return value
