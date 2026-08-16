from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAGE = ROOT / "app/widgets/reconstruction_page.py"
MAIN = ROOT / "app/main_window.py"
RESOURCE = ROOT / "app/reconstruction_digital_twin_resources.py"


def _imports(path: Path) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            values.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            values.append(node.module or "")
    return tuple(values)


def test_96g_resource_coordinator_stays_dependency_light_and_external_runtime_safe() -> None:
    forbidden = ("torch", "nerfstudio", "gsplat", "pycolmap", "depth_anything_3")
    imports = _imports(RESOURCE) + _imports(PAGE) + _imports(MAIN)
    assert not any(any(name.startswith(prefix) for prefix in forbidden) for name in imports)
    source = RESOURCE.read_text(encoding="utf-8")
    assert "static previews are not runtime owners" in source.lower()
    assert "canonical_geometry_authoritative: bool = False" in source
    assert "certified_metrology: bool = False" in source


def test_96g_page_centralizes_e2_e3_handoff_and_blocks_reverse_overlap() -> None:
    source = PAGE.read_text(encoding="utf-8")
    assert "def _release_appearance_resources_for_heavy_handoff" in source
    assert 'self._hide_interactive_nerf(blocking=True)' in source
    assert "Interactive NeRF resources did not release" in source
    assert 'self._release_appearance_resources_for_heavy_handoff(spec.title)' in source
    assert 'self._release_appearance_resources_for_heavy_handoff("Reconstruction")' in source
    assert 'self._interactive_resource_admission_failure("interactive NeRF")' in source
    assert 'self._interactive_resource_admission_failure("interactive Gaussian")' in source


def test_96g_application_resource_boundary_observes_page_owned_build_nerf_and_gaussian() -> None:
    source = MAIN.read_text(encoding="utf-8")
    assert "digital_twin_resource_activity_changed.connect" in source
    assert "self._reconstruction_page.digital_twin_resource_active" in source
    assert "Reconstruction Digital Twin resources are active" in source
    activity = source[source.index("    def _on_reconstruction_activity_changed"):source.index("    def _on_mode_changed")]
    assert "self._reconstruction_page.digital_twin_resource_active" in activity


def test_96g_processing_ui_surfaces_owner_and_startup_gpu_snapshot_without_live_vram_claim() -> None:
    source = PAGE.read_text(encoding="utf-8")
    assert '"Resource owner", self._digital_twin_processing_resource_value' in source
    assert '"GPU snapshot", self._digital_twin_processing_gpu_value' in source
    assert "Startup hardware snapshot only" in source
    resource = RESOURCE.read_text(encoding="utf-8")
    assert "startup_resources_text" in resource
    assert "startup" in resource.lower()
    assert "torch.cuda" not in resource
