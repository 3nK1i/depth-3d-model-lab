from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from core.reconstruction.mvs_environment import (
    MVS_PREFLIGHT_SCHEMA_VERSION,
    PYCOLMAP_MVS_REQUIRED_APIS,
    MvsGpuResource,
    MvsHostResourceProbe,
    MvsPreflightReport,
    probe_pycolmap_mvs_environment,
)
import core.reconstruction.mvs_environment as mvs_environment


def test_external_mvs_probe_accepts_exact_cuda_pycolmap_capabilities(monkeypatch) -> None:
    payload = {
        "available": True,
        "pycolmap_version": "4.1.1",
        "distribution": "pycolmap-cuda12",
        "distribution_version": "4.1.1",
        "cuda_devices": 1,
        "apis": {name: True for name in PYCOLMAP_MVS_REQUIRED_APIS},
        "patch_match_options_constructible": True,
        "stereo_fusion_options_constructible": True,
        "patch_match_option_fields": ["geom_consistency", "gpu_index"],
        "stereo_fusion_option_fields": ["cache_size", "max_depth_error"],
        "error": "",
    }

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload) + "\n", stderr="")

    monkeypatch.setattr(mvs_environment.subprocess, "run", fake_run)
    probe = probe_pycolmap_mvs_environment("/opt/eia-pycolmap/bin/python")
    assert probe.mvs_api_ready
    assert probe.patchmatch_cuda_ready
    assert probe.recommended_version
    assert probe.cuda_distribution
    assert "gpu_index" in probe.patch_match_option_fields


def test_external_mvs_probe_fails_closed_when_patchmatch_api_is_missing(monkeypatch) -> None:
    apis = {name: True for name in PYCOLMAP_MVS_REQUIRED_APIS}
    apis["patch_match_stereo"] = False
    payload = {
        "available": True,
        "pycolmap_version": "4.1.1",
        "distribution": "pycolmap-cuda12",
        "distribution_version": "4.1.1",
        "cuda_devices": 1,
        "apis": apis,
        "patch_match_options_constructible": True,
        "stereo_fusion_options_constructible": True,
        "patch_match_option_fields": [],
        "stereo_fusion_option_fields": [],
        "error": "",
    }

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload) + "\n", stderr="")

    monkeypatch.setattr(mvs_environment.subprocess, "run", fake_run)
    probe = probe_pycolmap_mvs_environment("python")
    assert not probe.mvs_api_ready
    assert not probe.patchmatch_cuda_ready


def test_preflight_guardrails_do_not_promote_reconstruction_authority() -> None:
    env = mvs_environment.PycolmapMvsEnvironmentProbe(
        python_executable="/opt/eia-pycolmap/bin/python",
        available=True,
        pycolmap_version="4.1.1",
        distribution="pycolmap-cuda12",
        distribution_version="4.1.1",
        cuda_devices=1,
        apis=tuple((name, True) for name in PYCOLMAP_MVS_REQUIRED_APIS),
        patch_match_options_constructible=True,
        stereo_fusion_options_constructible=True,
    )
    resources = MvsHostResourceProbe(
        workspace_path="/tmp/eia_i4_1_9a",
        disk_free_gib=20.0,
        ram_available_gib=12.0,
        gpus=(MvsGpuResource(index=0, name="RTX", total_vram_gib=6.0, free_vram_gib=5.0),),
    )
    report = MvsPreflightReport(
        environment=env,
        resources=resources,
        min_free_disk_gib=8.0,
        min_available_ram_gib=4.0,
        min_free_vram_gib=2.0,
    )
    payload = report.as_dict()
    assert report.ready
    assert payload["schema_version"] == MVS_PREFLIGHT_SCHEMA_VERSION
    assert payload["read_only_preflight"] is True
    assert payload["heavy_mvs_started"] is False
    assert payload["sfm_rerun"] is False
    assert payload["bundle_adjustment_rerun"] is False
    assert payload["camera_geometry_modified"] is False
    assert payload["metric_scale_claimed"] is False
    assert payload["measurement_authoritative"] is False
    assert payload["certified_metrology"] is False
    assert payload["surface_mesh_generated"] is False
    assert "not production limits" in payload["guardrails"]["scope"]


def test_preflight_rejects_insufficient_resource_headroom() -> None:
    env = mvs_environment.PycolmapMvsEnvironmentProbe(
        python_executable="/opt/eia-pycolmap/bin/python",
        available=True,
        pycolmap_version="4.1.1",
        distribution="pycolmap-cuda12",
        distribution_version="4.1.1",
        cuda_devices=1,
        apis=tuple((name, True) for name in PYCOLMAP_MVS_REQUIRED_APIS),
        patch_match_options_constructible=True,
        stereo_fusion_options_constructible=True,
    )
    resources = MvsHostResourceProbe(
        workspace_path="/tmp/eia_i4_1_9a",
        disk_free_gib=7.0,
        ram_available_gib=3.0,
        gpus=(MvsGpuResource(index=0, name="RTX", total_vram_gib=6.0, free_vram_gib=1.5),),
    )
    report = MvsPreflightReport(environment=env, resources=resources)
    assert report.runtime_ready
    assert not report.resource_ready
    assert not report.ready


def test_resource_probe_parses_linux_meminfo_and_nvidia_smi(monkeypatch, tmp_path: Path) -> None:
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:       8388608 kB\nMemAvailable:   6291456 kB\n", encoding="utf-8")
    assert mvs_environment._available_ram_gib(meminfo) == 6.0

    def fake_run(*args, **kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout="0, NVIDIA GeForce RTX 2060, 6144, 5120\n",
            stderr="",
        )

    monkeypatch.setattr(mvs_environment.subprocess, "run", fake_run)
    gpus, error = mvs_environment._probe_nvidia_smi()
    assert error == ""
    assert len(gpus) == 1
    assert gpus[0].total_vram_gib == 6.0
    assert gpus[0].free_vram_gib == 5.0
