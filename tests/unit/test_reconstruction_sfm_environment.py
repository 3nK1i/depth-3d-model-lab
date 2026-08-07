from __future__ import annotations

import json
from types import SimpleNamespace

from core.reconstruction import (
    PYCOLMAP_REQUIRED_APIS,
    probe_pycolmap_environment,
)
import core.reconstruction.sfm_environment as sfm_environment


def test_environment_probe_parses_external_pycolmap_capabilities(monkeypatch) -> None:
    payload = {
        "available": True,
        "pycolmap_version": "4.1.1",
        "distribution": "pycolmap-cuda12",
        "distribution_version": "4.1.1",
        "cuda_devices": 1,
        "apis": {name: True for name in PYCOLMAP_REQUIRED_APIS},
        "error": "",
    }

    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload) + "\n", stderr="")

    monkeypatch.setattr(sfm_environment.subprocess, "run", fake_run)
    probe = probe_pycolmap_environment("/opt/reconstruction/bin/python")
    assert probe.backend_ready
    assert probe.cuda_ready
    assert probe.recommended_version
    assert probe.distribution == "pycolmap-cuda12"


def test_environment_probe_failure_does_not_import_pycolmap_here(monkeypatch) -> None:
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps({
            "available": False,
            "pycolmap_version": "",
            "distribution": "",
            "distribution_version": "",
            "cuda_devices": 0,
            "apis": {},
            "error": "ModuleNotFoundError: pycolmap",
        }) + "\n", stderr="")

    monkeypatch.setattr(sfm_environment.subprocess, "run", fake_run)
    probe = probe_pycolmap_environment("python")
    assert not probe.available
    assert not probe.backend_ready
    assert "pycolmap" in probe.error
