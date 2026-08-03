"""MODEL card integration for hardware profiles and adapter selection."""

from __future__ import annotations

from app.widgets.settings_panel import SettingsPanel
from core.hardware.gpu_info import AcceleratorBackend, HardwareInfo
from core.hardware.profiles import ComputeProfile
from core.models.catalog import MOGE2_MODEL_ID, MOCK_MODEL_ID


class _MemorySettings:
    def __init__(
        self,
        profile: str | None = None,
        model_id: str | None = None,
    ) -> None:
        self.profile = profile
        self.model_id = model_id

    def load_compute_profile(self) -> str | None:
        return self.profile

    def save_compute_profile(self, profile: str) -> None:
        self.profile = profile

    def load_last_model_id(self) -> str | None:
        return self.model_id

    def save_last_model_id(self, model_id: str) -> None:
        self.model_id = model_id


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


def test_2060_auto_profile_is_efficient_and_moge_is_default(qtbot) -> None:
    panel = SettingsPanel(
        hardware_info=_gpu("NVIDIA GeForce RTX 2060", 6.0, (7, 5)),
        app_settings=_MemorySettings(),
    )
    qtbot.addWidget(panel)

    assert "RTX 2060" in panel._gpu_profile.itemText(0)
    assert panel.selected_model_id == MOGE2_MODEL_ID
    assert panel._model_combo.currentText() == "MoGe-2"
    assert panel._model_combo.isEnabled()
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
    assert panel.model_runtime_config["warmup_size"] == 512
    assert panel.model_runtime_config["warmup_runs"] == 2
    assert panel.model_runtime_config["allow_tf32"] is True


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


def test_mock_selection_is_persisted_and_marked_development_only(qtbot) -> None:
    settings = _MemorySettings(model_id=MOCK_MODEL_ID)
    panel = SettingsPanel(
        hardware_info=_gpu("NVIDIA GeForce RTX 2060", 6.0, (7, 5)),
        app_settings=settings,
    )
    qtbot.addWidget(panel)

    assert panel.selected_model_id == MOCK_MODEL_ID
    assert panel._model_badge.text() == "DEV ONLY"
    panel.set_model_loading()
    assert not panel._model_combo.isEnabled()
    panel.set_model_unloaded()
    assert panel._model_combo.isEnabled()


def test_5090_auto_profile_downgrades_under_startup_memory_pressure(qtbot) -> None:
    hardware = _gpu("NVIDIA GeForce RTX 5090", 32.0, (12, 0))
    pressured = HardwareInfo(
        backend=hardware.backend,
        device=hardware.device,
        device_index=hardware.device_index,
        name=hardware.name,
        total_memory_bytes=hardware.total_memory_bytes,
        free_memory_bytes=int(10.0 * 1024**3),
        compute_capability=hardware.compute_capability,
        supports_fp16=hardware.supports_fp16,
        supports_bf16=hardware.supports_bf16,
        torch_available=hardware.torch_available,
        torch_cuda_available=hardware.torch_cuda_available,
        torch_version=hardware.torch_version,
        torch_cuda_version=hardware.torch_cuda_version,
        compiled_architectures=hardware.compiled_architectures,
        torch_build_compatible=hardware.torch_build_compatible,
    )
    panel = SettingsPanel(hardware_info=pressured, app_settings=_MemorySettings())
    qtbot.addWidget(panel)

    assert panel.resolved_profile.effective is ComputeProfile.BALANCED
    assert panel.model_runtime_config["pretrained"] == "Ruicheng/moge-2-vitb-normal"
    assert panel.live_point_budget == 250_000
