from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "app" / "reconstruction_appearance_resources.py"
PAGE = ROOT / "app" / "widgets" / "reconstruction_page.py"
F1 = ROOT / "tests" / "unit" / "test_phase8_8f1_contract.py"
F2 = ROOT / "tests" / "unit" / "test_phase8_8f2_contract.py"
F3 = ROOT / "tests" / "unit" / "test_phase8_8f3_contract.py"
F4 = ROOT / "tests" / "unit" / "test_phase8_8f4_contract.py"


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    result: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.append(node.module or "")
    return result


def test_f5_resource_layer_uses_existing_hardware_snapshot_without_heavy_backends() -> None:
    imports = _imports(MODULE) + _imports(PAGE)
    forbidden = ("nerfstudio", "tinycudann", "torch", "gsplat", "pycolmap")
    assert not any(any(value.startswith(prefix) for prefix in forbidden) for value in imports)
    combined = MODULE.read_text(encoding="utf-8") + PAGE.read_text(encoding="utf-8")
    for call in (
        "subprocess.run(",
        "subprocess.Popen(",
        "run_gaussian_external_worker(",
        "run_nerf_external_worker(",
        "torch.load(",
    ):
        assert call not in combined
    assert "detect_hardware" in MODULE.read_text(encoding="utf-8")


def test_f5_adds_one_compact_resource_fallback_group_without_duplicate_selector() -> None:
    page = PAGE.read_text(encoding="utf-8")
    assert page.count('self._appearance_resource_group = self._settings_group("RESOURCE & FALLBACK")') == 1
    start = page.index('self._appearance_resource_group = self._settings_group("RESOURCE & FALLBACK")')
    end = page.index('authority = self._settings_group("AUTHORITY & SAFETY")', start)
    group = page[start:end]
    for label in ("Device", "Resources", "Presentation", "Fallback", "Pipeline", "State"):
        assert f'"{label}"' in group
    assert "QComboBox()" not in group
    assert "QPushButton(" not in group


def test_f5_static_preview_is_not_gated_by_cuda_and_fallback_is_canonical_only() -> None:
    module = MODULE.read_text(encoding="utf-8")
    assert "static previews remain supported" in module
    assert "Canonical geometry • no neural cross-fallback" in module
    assert "cross_representation_fallback" in module
    assert 'object.__setattr__(self, "cross_representation_fallback", False)' in module
    assert "fallback_to" not in module


def test_f5_surfaces_persistent_appearance_fallback_evidence_without_rerunning_pipeline() -> None:
    module = MODULE.read_text(encoding="utf-8")
    assert "fallback_evidence" in module
    assert '"gaussian_appearance"' in module
    assert '"nerf_appearance"' in module
    assert "ReconstructionPipelineOrchestrator" not in module
    assert "execute(" not in module


def test_f5_preserves_authority_and_is_object_agnostic() -> None:
    combined = (MODULE.read_text(encoding="utf-8") + PAGE.read_text(encoding="utf-8")).lower()
    for token in (
        "connector_body",
        "connector_pin",
        "missing_pin",
        "bent_pin",
        'class_name == "gear"',
        'class_name == "housing"',
        "semantic_role ==",
    ):
        assert token not in combined
    assert "canonical geometry remains authoritative" in combined
    assert "certified metrology" in MODULE.read_text(encoding="utf-8").lower()


def test_prior_f1_to_f4_contracts_advance_to_f5_without_reopening_backend_scope() -> None:
    assert "resource/fallback UX available in F5" in F1.read_text(encoding="utf-8")
    assert "F5 reports resource state and canonical fallback" in F2.read_text(encoding="utf-8")
    assert "F5 reports resource state and canonical fallback" in F3.read_text(encoding="utf-8")
    f4 = F4.read_text(encoding="utf-8")
    assert "F5 reports the fallback reason" in f4
    assert "resource_probe" not in (ROOT / "app" / "reconstruction_representation_switching.py").read_text(encoding="utf-8")
