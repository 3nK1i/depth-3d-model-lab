from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/run_phase8_9f6_actual_application_acceptance.py"
SPEC = importlib.util.spec_from_file_location("phase8_9f6_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def _f5_payload(tmp_path: Path) -> dict:
    manifest = tmp_path / "scene-a/persistent_reconstruction_scene.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}", encoding="utf-8")
    return {
        "phase": "8.9F5",
        "pass": True,
        "blockers": [],
        "authority": {
            "canonical_geometry": "authoritative",
            "gaussian": "appearance_only",
            "nerf": "appearance_only",
            "certified_metrology": False,
        },
        "scene": {
            "state": "ready",
            "canonical_geometry_authoritative": True,
            "manifest": str(manifest),
        },
        "presentation_readiness": {
            "gaussian": {"ready": True},
            "nerf": {"ready": True},
        },
    }


def test_f6_contract_is_validation_only_and_pins_clean_f5_baseline() -> None:
    assert mod.PHASE == "8.9F6"
    assert mod.EXPECTED_BRANCH == "phase8-reconstruction"
    assert mod.EXPECTED_HEAD == "eb5062bebeb146ad98d658c19fab7ec4750f2834"
    assert len(mod.F6_REPO_FILES) == 5
    assert all(not path.startswith(("app/", "core/", "config/")) for path in mod.F6_REPO_FILES)


def test_f6_accepts_qualified_f5_authority_and_preview_readiness(tmp_path: Path) -> None:
    payload = _f5_payload(tmp_path)
    result = mod._validate_f5_summary(payload, tmp_path)
    assert result["manifest"].is_file()
    assert all(result["checks"].values())


@pytest.mark.parametrize(
    "path,value",
    [
        (("pass",), False),
        (("authority", "gaussian"), "canonical_geometry"),
        (("authority", "nerf"), "canonical_geometry"),
        (("authority", "certified_metrology"), True),
        (("presentation_readiness", "gaussian", "ready"), False),
        (("presentation_readiness", "nerf", "ready"), False),
    ],
)
def test_f6_rejects_invalid_f5_source_claims(tmp_path: Path, path: tuple[str, ...], value) -> None:
    payload = _f5_payload(tmp_path)
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(RuntimeError, match="F5 source qualification failed"):
        mod._validate_f5_summary(payload, tmp_path)


@dataclass(frozen=True)
class _Artifact:
    artifact_id: str
    kind: object
    payload_snapshot: dict

    def as_dict(self):
        return {
            "artifact_id": self.artifact_id,
            "kind": getattr(self.kind, "value", self.kind),
            "payload_snapshot": dict(self.payload_snapshot),
        }


def test_f6_fallback_clone_never_mutates_real_scene_artifacts(tmp_path: Path) -> None:
    gaussian = _Artifact("g", SimpleNamespace(value="gaussian_scene"), {"artifact_dir": "/real/g"})
    nerf = _Artifact("n", SimpleNamespace(value="nerf_scene"), {"artifact_dir": "/real/n", "preview_path": "/real/n/preview.png"})
    scene = SimpleNamespace(
        scene_id="prs-test",
        request_id="r",
        session_id="00000000-0000-4000-8000-000000000001",
        state="ready",
        pipeline_status="completed",
        artifacts=(gaussian, nerf),
        provider_provenance=(),
        fallback_evidence=(),
        file_references=(),
        canonical_geometry_artifact_id="canonical",
        scene_fingerprint_sha256="0" * 64,
        created_at_utc="2026-08-12T00:00:00+00:00",
    )
    clone = mod._fallback_scene_clone(scene, tmp_path / "missing")
    assert nerf.payload_snapshot["artifact_dir"] == "/real/n"
    assert clone.nerf_artifact.payload_snapshot != nerf.payload_snapshot
    assert str(tmp_path / "missing") in dict(clone.nerf_artifact.payload_snapshot)["artifact_dir"]
    assert clone.gaussian_artifact is gaussian


def test_f6_script_does_not_import_or_launch_heavy_backends_directly() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    forbidden_imports = (
        "import pycolmap",
        "import gsplat",
        "import nerfstudio",
        "import depth_anything_3",
        "from pycolmap",
        "from gsplat",
        "from nerfstudio",
        "from depth_anything_3",
    )
    assert not any(token in source for token in forbidden_imports)
    assert "gaussian_training_started\": False" in source
    assert "nerfacto_training_started\": False" in source
    assert "pycolmap_started\": False" in source
    assert "da3_started\": False" in source


def test_f6_campaign_uses_modetabs_active_mode_property_not_callable() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert ".active_mode()" not in source
    assert ".active_mode is AppMode.RECONSTRUCTION" in source
