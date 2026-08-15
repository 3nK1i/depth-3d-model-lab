from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "app/widgets/reconstruction_page.py"
CLIENT = ROOT / "app/widgets/reconstruction_nerf_worker_client.py"
WORKER = ROOT / "scripts/nerf_workers/nerfstudio_interactive_worker.py"
CANVAS = ROOT / "app/widgets/viewport/vispy_canvas.py"


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            values.append(node.module or "")
    return tuple(values)


def test_5e_ui_process_does_not_import_heavy_nerf_runtime() -> None:
    forbidden = ("nerfstudio", "tinycudann", "torch")
    imports = _imports(PAGE) + _imports(CLIENT) + _imports(CANVAS)
    assert not any(any(value.startswith(prefix) for prefix in forbidden) for value in imports)
    worker_imports = _imports(WORKER)
    # The isolated worker is the only production path allowed to touch the heavy backend.
    assert any(value.startswith("nerfstudio") for value in worker_imports)


def test_5e_ui_exposes_explicit_interactive_preview_hide_and_canonical_measurement_guard() -> None:
    page = PAGE.read_text(encoding="utf-8")
    assert 'QPushButton("Interactive")' in page
    assert 'QPushButton("Preview")' in page
    assert 'QPushButton("Hide")' in page
    assert "Canonical measurement active • NeRF appearance paused" in page
    assert "turntable_state_to_nerf_render_request" in page
    assert "self._appearance_preview.set_mouse_passthrough(True)" in page
    assert "canonical geometry remains picking/measurement authority" in page.lower()


def test_5e_external_process_client_uses_process_group_cleanup_without_shell() -> None:
    source = CLIENT.read_text(encoding="utf-8")
    assert 'shutil.which("setsid")' in source
    assert "os.killpg" in source
    assert "signal.SIGTERM" in source
    assert "signal.SIGKILL" in source
    assert "QProcess" in source
    assert "shell=True" not in source


def test_5e_real_target_smoke_integrity_verifies_persistent_nerf_before_backend_start() -> None:
    source = (ROOT / "scripts/run_phase9_5e_real_interactive_render.py").read_text(encoding="utf-8")
    assert "verify_interactive_nerf_integrity(scene, presentation)" in source
    assert source.index("verify_interactive_nerf_integrity(scene, presentation)") < source.index("subprocess.Popen(")


def test_5e_worker_stages_checkpoint_because_nerfstudio_eval_setup_recomputes_load_dir() -> None:
    source = WORKER.read_text(encoding="utf-8")
    assert "isolated_staged_rebind_from_immutable_training_config" in source
    assert "config.get_checkpoint_dir()" in source
    assert "config.output_dir = staging_root" in source
    assert "os.link(checkpoint_path, staged_checkpoint)" in source
    assert "shutil.copy2(checkpoint_path, staged_checkpoint)" in source
