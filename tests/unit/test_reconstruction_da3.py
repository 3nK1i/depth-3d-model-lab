from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest

from core.reconstruction.da3 import (
    DA3_ALLOWED_MODEL_IDS, DA3_BALANCED_MODEL_ID, DA3_EFFICIENT_MODEL_ID,
    DA3_GEOMETRY_SCALE, DA3_INTERNAL_POSE_CONVENTION, DA3_RUNTIME_CUDA,
    DA3_RUNTIME_TORCH, DA3_RUNTIME_TORCHVISION, DA3_RUNTIME_XFORMERS,
    DA3EnvironmentProbe, DA3ResultManifest, DA3RuntimeLock,
    da3_opencv_w2c_to_reconstruction_from_camera, read_da3_result_manifest,
    validate_da3_depth_confidence, validate_da3_intrinsics, validate_da3_npz_result,
)
from core.reconstruction.errors import ReconstructionValidationError


def test_f2_model_allowlist_is_small_and_base_only() -> None:
    assert DA3_ALLOWED_MODEL_IDS == (DA3_EFFICIENT_MODEL_ID, DA3_BALANCED_MODEL_ID)


def test_runtime_versions_are_explicit() -> None:
    assert DA3_RUNTIME_TORCH == "2.7.1"
    assert DA3_RUNTIME_TORCHVISION == "0.22.1"
    assert DA3_RUNTIME_XFORMERS == "0.0.31"
    assert DA3_RUNTIME_CUDA == "12.8"


def test_runtime_lock_accepts_expected_stack() -> None:
    lock = DA3RuntimeLock("a"*40, "b"*64, "3.11.9", "2.7.1+cu128", "0.22.1+cu128", "0.0.31", "12.8")
    assert lock.expected_versions_match


def test_runtime_lock_rejects_bad_revision() -> None:
    with pytest.raises(ReconstructionValidationError):
        DA3RuntimeLock("bad", "b"*64, "3.11.9", "2.7.1", "0.22.1", "0.0.31", "12.8")


def test_runtime_lock_detects_version_drift() -> None:
    lock = DA3RuntimeLock("a"*40, "b"*64, "3.11.9", "2.8.0", "0.22.1", "0.0.31", "12.8")
    assert not lock.expected_versions_match


def test_environment_probe_availability_contract(tmp_path: Path) -> None:
    probe = DA3EnvironmentProbe(
        python_executable="/usr/bin/python3", repository_root=str(tmp_path.resolve()),
        source_revision="a"*40, python_version="3.11.1", torch_version="2.7.1+cu128",
        torchvision_version="0.22.1+cu128", xformers_version="0.0.31", torch_cuda_version="12.8",
        cuda_available=True, device_names=("RTX",), device_capabilities=((7,5),),
        depth_anything_3_importable=True, turing_compatibility_patch_applied=True, runtime_lock_valid=True,
    )
    assert probe.available


def test_environment_probe_requires_patch(tmp_path: Path) -> None:
    probe = DA3EnvironmentProbe(
        python_executable="/usr/bin/python3", repository_root=str(tmp_path.resolve()), source_revision="a"*40,
        cuda_available=True, device_names=("RTX",), device_capabilities=((7,5),), depth_anything_3_importable=True,
        turing_compatibility_patch_applied=False, runtime_lock_valid=True,
    )
    assert not probe.available


def test_w2c_identity_converts_to_identity_camera_to_world() -> None:
    w2c = np.eye(4)[:3]
    pose = da3_opencv_w2c_to_reconstruction_from_camera(w2c)
    assert pose.shape == (1,4,4)
    np.testing.assert_allclose(pose[0], np.eye(4))


def test_w2c_translation_is_inverted() -> None:
    w2c = np.eye(4); w2c[:3,3] = [1,2,3]
    pose = da3_opencv_w2c_to_reconstruction_from_camera(w2c)
    np.testing.assert_allclose(pose[0,:3,3], [-1,-2,-3])


def test_w2c_rejects_nonrigid_rotation() -> None:
    bad = np.eye(4); bad[0,0]=2
    with pytest.raises(ReconstructionValidationError):
        da3_opencv_w2c_to_reconstruction_from_camera(bad)


def test_intrinsics_validation() -> None:
    K=np.array([[[100,0,20],[0,100,10],[0,0,1]]],dtype=float)
    out=validate_da3_intrinsics(K,frame_count=1)
    assert out.shape==(1,3,3) and not out.flags.writeable


def test_intrinsics_reject_negative_focal() -> None:
    K=np.array([[[-1,0,20],[0,100,10],[0,0,1]]],dtype=float)
    with pytest.raises(ReconstructionValidationError): validate_da3_intrinsics(K)


def test_depth_confidence_validation() -> None:
    d,c=validate_da3_depth_confidence(np.ones((2,4,5)),np.ones((2,4,5))*0.5)
    assert d.shape==(2,4,5) and not d.flags.writeable and not c.flags.writeable


def test_depth_rejects_nonpositive() -> None:
    d=np.ones((1,2,2)); d[0,0,0]=0
    with pytest.raises(ReconstructionValidationError): validate_da3_depth_confidence(d,np.ones_like(d))


def _manifest(tmp_path: Path) -> DA3ResultManifest:
    npz=tmp_path/"r.npz"
    np.savez_compressed(npz, depth=np.ones((2,4,5),np.float32), confidence=np.ones((2,4,5),np.float32), extrinsics_w2c=np.tile(np.eye(4,dtype=np.float32)[None,:3,:],(2,1,1)), intrinsics=np.tile(np.array([[10,0,2],[0,10,2],[0,0,1]],dtype=np.float32)[None],(2,1,1)))
    return DA3ResultManifest("req",DA3_EFFICIENT_MODEL_ID,"c"*40,"d"*40,str(npz.resolve()),2,5,4,392,1.2,512,"RTX")


def test_result_manifest_keeps_da3_noncanonical(tmp_path: Path) -> None:
    m=_manifest(tmp_path)
    assert m.geometry_scale==DA3_GEOMETRY_SCALE
    assert not m.canonical_geometry_authoritative and not m.certified_metrology
    assert m.as_dict()["internal_pose_convention"]==DA3_INTERNAL_POSE_CONVENTION


def test_result_manifest_rejects_metric_authority_claim(tmp_path: Path) -> None:
    m=_manifest(tmp_path).as_dict(); m["certified_metrology"]=True
    with pytest.raises(ReconstructionValidationError): DA3ResultManifest.from_payload(m)


def test_result_manifest_roundtrip(tmp_path: Path) -> None:
    m=_manifest(tmp_path); p=tmp_path/"manifest.json"; p.write_text(json.dumps(m.as_dict()))
    assert read_da3_result_manifest(p)==m


def test_validate_npz_normalizes_pose(tmp_path: Path) -> None:
    m=_manifest(tmp_path); arrays=validate_da3_npz_result(m)
    assert arrays["T_reconstruction_from_camera"].shape==(2,4,4)
    assert arrays["depth"].shape==(2,4,5)


def test_validate_npz_rejects_missing_array(tmp_path: Path) -> None:
    m=_manifest(tmp_path); np.savez_compressed(m.output_npz_path, depth=np.ones((2,4,5)))
    with pytest.raises(ReconstructionValidationError): validate_da3_npz_result(m)
