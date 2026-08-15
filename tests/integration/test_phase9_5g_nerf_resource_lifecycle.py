from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "app/widgets/reconstruction_page.py"
CLIENT = ROOT / "app/widgets/reconstruction_nerf_worker_client.py"
WORKER = ROOT / "scripts/nerf_workers/nerfstudio_interactive_worker.py"
RESOURCE = ROOT / "app/reconstruction_nerf_resources.py"


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            values.append(node.module or "")
    return tuple(values)


def test_5g_production_resource_path_keeps_heavy_nerf_imports_external() -> None:
    forbidden = ("nerfstudio", "tinycudann", "torch")
    imports = _imports(PAGE) + _imports(CLIENT) + _imports(RESOURCE)
    assert not any(any(name.startswith(prefix) for prefix in forbidden) for name in imports)
    worker_imports = _imports(WORKER)
    assert any(name.startswith("nerfstudio") for name in worker_imports)


def test_5g_client_bounds_render_queue_and_preserves_process_group_cleanup() -> None:
    source = CLIENT.read_text(encoding="utf-8")
    assert "LatestRenderRequestQueue" in source
    assert 'return "coalesced"' in source
    assert "self._render_queue.complete" in source
    assert "self._render_queue.cancel_pending" in source
    assert 'shutil.which("setsid")' in source
    assert "os.killpg" in source
    assert "signal.SIGTERM" in source
    assert "signal.SIGKILL" in source
    assert "shell=True" not in source


def test_5g_worker_reports_real_cuda_telemetry_and_bounds_frame_residency() -> None:
    source = WORKER.read_text(encoding="utf-8")
    assert "torch.cuda.memory_allocated" in source
    assert "torch.cuda.memory_reserved" in source
    assert "torch.cuda.max_memory_allocated" in source
    assert "MAX_RETAINED_FRAMES = 2" in source
    assert "_prune_frames" in source
    assert '"resource_hardening_phase": "9.5G"' in source
    assert '"scene_role": "appearance_only"' in source


def test_5g_page_releases_worker_on_hide_switch_and_shutdown_and_labels_telemetry() -> None:
    source = PAGE.read_text(encoding="utf-8")
    assert 'self._add_status_row(nerf_layout, "Resources", self._nerf_resources_value)' in source
    assert 'self._add_status_row(nerf_layout, "Performance", self._nerf_performance_value)' in source
    assert 'self._add_status_row(nerf_layout, "Lifecycle", self._nerf_lifecycle_value)' in source
    assert "_nerf_release_pending = True" in source
    assert "self._nerf_worker_client.stop_blocking()" in source
    assert "self._hide_nerf_visualization()" in source
    assert "self._hide_interactive_nerf(blocking=True)" in source
    assert "CUDA allocation/reservation values come from torch inside the isolated Nerfstudio worker" in source


def test_5g_real_target_check_measures_worker_pid_memory_and_proves_release() -> None:
    source = (ROOT / "scripts/run_phase9_5g_real_nerf_resource_check.py").read_text(encoding="utf-8")
    assert '"--query-compute-apps=pid,used_memory"' in source
    assert "_wait_released" in source
    assert "pid not in gpu_processes" in source
    assert '"all_worker_pids_released_from_os_and_nvidia_smi"' in source
    assert '"max_nvidia_smi_worker_memory_mib"' in source
    assert '"render_timings_are_observation_only": True' in source
    assert '"training_reexecuted": False' in source
    assert '"persistence_modified": False' in source
