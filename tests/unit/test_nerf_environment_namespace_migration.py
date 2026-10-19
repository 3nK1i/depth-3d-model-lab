from __future__ import annotations

import importlib
from pathlib import Path

import pytest


CHECKOUT_WORKERS = (
    "scripts.nerf_workers.nerfstudio_worker",
    "scripts.nerf_workers.nerfstudio_e3_worker",
    "scripts.nerf_workers.nerfstudio_e4_worker",
)

PUBLIC_ENVIRONMENT_NAMES = (
    "DEPTH3D_NERFSTUDIO_ROOT",
    "DEPTH3D_TCNN_ROOT",
)


@pytest.mark.parametrize("module_name", CHECKOUT_WORKERS)
@pytest.mark.parametrize("public_name", PUBLIC_ENVIRONMENT_NAMES)
def test_checkout_workers_use_public_environment_namespace(
    monkeypatch,
    tmp_path: Path,
    module_name: str,
    public_name: str,
) -> None:
    module = importlib.import_module(module_name)

    root = tmp_path / "checkout"
    (root / ".git").mkdir(parents=True)

    monkeypatch.setenv(public_name, str(root))
    monkeypatch.setattr(
        module,
        "_git_revision",
        lambda _path: "abc123",
    )

    resolved, revision = module._require_checkout(
        public_name,
        "abc",
    )

    assert resolved == root.resolve()
    assert revision == "abc123"


@pytest.mark.parametrize(
    "public_name",
    PUBLIC_ENVIRONMENT_NAMES,
)
def test_interactive_worker_uses_public_environment_namespace(
    monkeypatch,
    public_name: str,
) -> None:
    module = importlib.import_module(
        "scripts.nerf_workers.nerfstudio_interactive_worker"
    )

    monkeypatch.setenv(
        public_name,
        "/public/backend",
    )

    assert (
        module._environment_value(public_name)
        == "/public/backend"
    )


def test_worker_call_sites_use_public_namespace() -> None:
    root = Path(__file__).resolve().parents[2]

    for relative in (
        "scripts/nerf_workers/nerfstudio_worker.py",
        "scripts/nerf_workers/nerfstudio_e3_worker.py",
        "scripts/nerf_workers/nerfstudio_e4_worker.py",
        "scripts/nerf_workers/nerfstudio_interactive_worker.py",
    ):
        text = (
            root / relative
        ).read_text(encoding="utf-8")

        assert "DEPTH3D_NERFSTUDIO_ROOT" in text
        assert "DEPTH3D_TCNN_ROOT" in text


def test_nerf_build_producer_emits_public_environment_namespace(
    tmp_path: Path,
) -> None:
    from app.reconstruction_nerf_build import (
        _isolated_environment_overrides,
    )

    prefix = tmp_path / "nerf-runtime"
    python = prefix / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"python")

    nerfstudio_root = tmp_path / "nerfstudio"
    tcnn_root = tmp_path / "tiny-cuda-nn"
    nerfstudio_root.mkdir()
    tcnn_root.mkdir()

    overrides = dict(
        _isolated_environment_overrides(
            python,
            nerfstudio_root=nerfstudio_root,
            tcnn_root=tcnn_root,
        )
    )

    assert (
        overrides["DEPTH3D_NERFSTUDIO_ROOT"]
        == str(nerfstudio_root)
    )
    assert (
        overrides["DEPTH3D_TCNN_ROOT"]
        == str(tcnn_root)
    )


def test_interactive_client_emits_public_namespace() -> None:
    root = Path(__file__).resolve().parents[2]

    text = (
        root / "app/widgets/reconstruction_nerf_worker_client.py"
    ).read_text(encoding="utf-8")

    assert 'env["DEPTH3D_NERFSTUDIO_ROOT"]' in text
    assert 'env["DEPTH3D_TCNN_ROOT"]' in text


def test_qualification_helpers_use_public_nerf_environment_namespace() -> None:
    root = Path(__file__).resolve().parents[2]

    helpers = (
        "scripts/nerf_e4_support.py",
        "scripts/reconstruction_engine_checkpoint_worker.py",
        "scripts/run_nerf_backend_qualification.py",
        "scripts/run_nerf_real_scene_validation.py",
        "scripts/run_phase9_5e_real_interactive_render.py",
        "scripts/run_phase9_5g_real_nerf_resource_check.py",
        "scripts/run_reconstruction_f4_real_backends.py",
    )

    for relative in helpers:
        source = (
            root / relative
        ).read_text(encoding="utf-8")

        assert "DEPTH3D_NERFSTUDIO_ROOT" in source
        assert "DEPTH3D_TCNN_ROOT" in source


def test_e4_worker_environment_emits_public_nerf_namespace(
    tmp_path: Path,
) -> None:
    from scripts.nerf_e4_support import worker_environment

    nerfstudio_root = tmp_path / "nerfstudio"
    tcnn_root = tmp_path / "tiny-cuda-nn"

    env = worker_environment(
        nerfstudio_root=nerfstudio_root,
        tcnn_root=tcnn_root,
    )

    assert (
        env["DEPTH3D_NERFSTUDIO_ROOT"]
        == str(nerfstudio_root)
    )
    assert (
        env["DEPTH3D_TCNN_ROOT"]
        == str(tcnn_root)
    )
