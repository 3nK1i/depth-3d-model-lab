"""MODEL card integration for the hardware-aware profile resolver."""

from __future__ import annotations

from core.hardware.gpu_info import AcceleratorBackend, HardwareInfo
from core.hardware.profiles import ComputeProfile
from app.widgets.settings_panel import SettingsPanel


class _MemorySettings:
    def __init__(self, profile: str | None = None) -> None:
        self.profile = profile

    def load_compute_profile(self) -> str | None:
        return self.profile

    def save_compute_profile(self, profile: str) -> None:
        self.profile = profile


def _gpu(name: str, vram_gib: float, capability: tuple[int, int]) -> HardwareInfo:
    return HardwareInfo(
        backend=AcceleratorBackend.CUDA,
        device="cuda:0",
        device_index=0,
        name=name,
        total_memory_bytes=int(vram_gib * 1024**3),
        free_memory_bytes=int(vram_gib * 0.9 * 1024**3),
        compute_capability=capability,
        supports_fp16=True,
        supports_bf16=capability[0] >= 8,
        torch_available=True,
        torch_cuda_available=True,
        torch_version="test",
        torch_cuda_version="test",
        compiled_architectures=(f"sm_{capability[0]}{capability[1]}",),
        torch_build_compatible=True,
    )


def test_2060_auto_profile_is_efficient(qtbot) -> None:
    panel = SettingsPanel(
        hardware_info=_gpu("NVIDIA GeForce RTX 2060", 6.0, (7, 5)),
        app_settings=_MemorySettings(),
    )
    qtbot.addWidget(panel)

    assert "RTX 2060" in panel._gpu_profile.itemText(0)
    assert panel._model_combo.currentText() == "Development Mock"
    assert not panel._model_combo.isEnabled()
    assert panel._model_variant.currentText() == "MoGe-2 ViT-S Normal"
    assert panel.resolved_profile.effective is ComputeProfile.EFFICIENT
    assert panel.model_runtime_config["pretrained"] == "Ruicheng/moge-2-vits-normal"
    assert panel.point_budget == 250_000
    assert panel.live_point_budget == 100_000


def test_5090_auto_profile_is_ultra(qtbot) -> None:
    panel = SettingsPanel(
        hardware_info=_gpu("NVIDIA GeForce RTX 5090", 32.0, (12, 0)),
        app_settings=_MemorySettings(),
    )
    qtbot.addWidget(panel)

    assert panel.resolved_profile.effective is ComputeProfile.ULTRA
    assert panel.model_runtime_config["pretrained"] == "Ruicheng/moge-2-vitl-normal"
    assert panel.point_budget == 1_000_000
    assert panel.live_point_budget == 500_000


def test_unsafe_manual_selection_uses_safe_fallback(qtbot) -> None:
    settings = _MemorySettings(ComputeProfile.ULTRA.value)
    panel = SettingsPanel(
        hardware_info=_gpu("NVIDIA GeForce RTX 2060", 6.0, (7, 5)),
        app_settings=settings,
    )
    qtbot.addWidget(panel)

    assert panel.resolved_profile.requested is ComputeProfile.ULTRA
    assert panel.resolved_profile.effective is ComputeProfile.EFFICIENT
    assert panel._model_badge.text() == "SAFE FALLBACK"
