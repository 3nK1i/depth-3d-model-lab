"""MODEL card integration for hardware profiles and adapter selection."""

from __future__ import annotations

from app.widgets.settings_panel import SettingsPanel
from core.hardware.gpu_info import AcceleratorBackend, HardwareInfo
from core.hardware.profiles import ComputeProfile
from core.models.catalog import (
    DEPTH_ANYTHING_V2_MODEL_ID,
    DEPTH_PRO_MODEL_ID,
    MOGE2_MODEL_ID,
    MOCK_MODEL_ID,
    METRIC3D_V2_MODEL_ID,
    UNIDEPTH_V2_MODEL_ID,
    VIDEO_DEPTH_ANYTHING_MODEL_ID,
    MARIGOLD_MODEL_ID,
)


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

def test_depth_pro_selection_uses_registry_profile_without_importing_model(qtbot) -> None:
    settings = _MemorySettings(model_id=DEPTH_PRO_MODEL_ID)
    panel = SettingsPanel(
        hardware_info=_gpu("NVIDIA GeForce RTX 2060", 6.0, (7, 5)),
        app_settings=settings,
    )
    qtbot.addWidget(panel)

    assert panel.selected_model_id == DEPTH_PRO_MODEL_ID
    assert panel._model_combo.currentText() == "Depth Pro"
    assert panel._gpu_profile.isEnabled()
    assert panel._model_variant.currentText() == "Depth Pro · DINOv2-L"
    assert panel._resolution.currentText() == "1536² internal · Metric"
    assert panel._precision.currentText() == "FP16"
    assert panel.model_runtime_config["checkpoint"] == "checkpoints/depth_pro.pt"
    assert panel.model_runtime_config["precision"] == "fp16"
    assert panel.model_runtime_config["device"] == "cuda:0"
    assert panel.point_budget == 250_000
    assert panel.live_point_budget == 100_000



def test_unidepth_v2_selection_uses_registry_profile_on_rtx2060(qtbot) -> None:
    panel = SettingsPanel(
        hardware_info=_gpu("NVIDIA GeForce RTX 2060", 6.0, (7, 5)),
        app_settings=_MemorySettings(model_id=UNIDEPTH_V2_MODEL_ID),
    )
    qtbot.addWidget(panel)

    assert panel.selected_model_id == UNIDEPTH_V2_MODEL_ID
    assert panel._model_combo.currentText() == "UniDepthV2"
    assert panel._gpu_profile.isEnabled()
    assert panel._model_variant.currentText() == "UniDepthV2 ViT-S/14"
    assert panel._resolution.currentText() == "Level 2 · Metric + uncertainty"
    assert panel._precision.currentText() == "FP16"
    assert panel.model_runtime_config["checkpoint_dir"] == "checkpoints/unidepth-v2/vits14"
    assert panel.model_runtime_config["variant"] == "vits14"
    assert panel.model_runtime_config["resolution_level"] == 2
    assert panel.point_budget == 250_000
    assert panel.live_point_budget == 100_000


def test_metric3d_v2_selection_uses_small_profile_on_rtx2060(qtbot) -> None:
    panel = SettingsPanel(
        hardware_info=_gpu("NVIDIA GeForce RTX 2060", 6.0, (7, 5)),
        app_settings=_MemorySettings(model_id=METRIC3D_V2_MODEL_ID),
    )
    qtbot.addWidget(panel)

    assert panel.selected_model_id == METRIC3D_V2_MODEL_ID
    assert panel._model_combo.currentText() == "Metric3D v2"
    assert panel._gpu_profile.isEnabled()
    assert panel._model_variant.currentText() == "Metric3D v2 ViT-S"
    assert panel._resolution.currentText() == "1064×616 · Metric + normals"
    assert panel._precision.currentText() == "FP16"
    assert panel.model_runtime_config["entrypoint"] == "metric3d_vit_small"
    assert panel.model_runtime_config["checkpoint"].endswith("metric_depth_vit_small_800k.pth")
    assert panel.point_budget == 250_000
    assert panel.live_point_budget == 100_000


def test_metric3d_v2_selection_uses_large_profile_on_rtx5090(qtbot) -> None:
    panel = SettingsPanel(
        hardware_info=_gpu("NVIDIA GeForce RTX 5090", 32.0, (12, 0)),
        app_settings=_MemorySettings(model_id=METRIC3D_V2_MODEL_ID),
    )
    qtbot.addWidget(panel)

    assert panel.selected_model_id == METRIC3D_V2_MODEL_ID
    assert panel._model_variant.currentText() == "Metric3D v2 ViT-L"
    assert panel.model_runtime_config["entrypoint"] == "metric3d_vit_large"
    assert panel.model_runtime_config["checkpoint"].endswith("metric_depth_vit_large_800k.pth")
    assert panel.point_budget == 1_000_000
    assert panel.live_point_budget == 500_000



def test_depth_anything_v2_selection_uses_small_profile_on_rtx2060(qtbot) -> None:
    panel = SettingsPanel(
        hardware_info=_gpu("NVIDIA GeForce RTX 2060", 6.0, (7, 5)),
        app_settings=_MemorySettings(model_id=DEPTH_ANYTHING_V2_MODEL_ID),
    )
    qtbot.addWidget(panel)
    assert panel._model_combo.currentText() == "Depth Anything V2"
    assert panel._model_variant.currentText() == "Depth Anything V2 Small"
    assert panel._resolution.currentText() == "518 px · Relative depth"
    assert panel._precision.currentText() == "FP16"
    assert panel.model_runtime_config["encoder"] == "vits"
    assert panel.model_runtime_config["input_size"] == 518
    assert panel.point_budget == 250_000
    assert panel.live_point_budget == 100_000


def test_video_depth_anything_selection_uses_streaming_small_profile_on_rtx2060(qtbot) -> None:
    panel = SettingsPanel(
        hardware_info=_gpu("NVIDIA GeForce RTX 2060", 6.0, (7, 5)),
        app_settings=_MemorySettings(model_id=VIDEO_DEPTH_ANYTHING_MODEL_ID),
    )
    qtbot.addWidget(panel)
    assert panel._model_combo.currentText() == "Video Depth Anything"
    assert panel._model_variant.currentText() == "Metric Video Depth Anything Small"
    assert panel._resolution.currentText() == "392 px · Temporal metric"
    assert panel._precision.currentText() == "FP16"
    assert panel.model_runtime_config["encoder"] == "vits"
    assert panel.model_runtime_config["max_resolution"] == 960
    assert panel.live_point_budget == 100_000


def test_marigold_selection_uses_fast_offload_profile_on_rtx2060(qtbot) -> None:
    panel = SettingsPanel(
        hardware_info=_gpu("NVIDIA GeForce RTX 2060", 6.0, (7, 5)),
        app_settings=_MemorySettings(model_id=MARIGOLD_MODEL_ID),
    )
    qtbot.addWidget(panel)
    assert panel._model_combo.currentText() == "Marigold"
    assert panel._model_variant.currentText() == "Marigold Depth v1.1 · Fast"
    assert panel._resolution.currentText() == "512 px · 1 step(s)"
    assert panel._precision.currentText() == "FP16"
    assert panel.model_runtime_config["num_inference_steps"] == 1
    assert panel.model_runtime_config["ensemble_size"] == 1
    assert panel.model_runtime_config["model_cpu_offload"] is True
    assert panel.model_runtime_config["inference_timeout_ms"] == 120_000
    assert panel.model_runtime_config["stop_timeout_ms"] == 120_000


def test_marigold_selection_uses_precision_uncertainty_profile_on_rtx5090(qtbot) -> None:
    panel = SettingsPanel(
        hardware_info=_gpu("NVIDIA GeForce RTX 5090", 32.0, (12, 0)),
        app_settings=_MemorySettings(model_id=MARIGOLD_MODEL_ID),
    )
    qtbot.addWidget(panel)
    assert panel._model_variant.currentText() == "Marigold Depth v1.1 · Precision"
    assert "uncertainty" in panel._resolution.currentText()
    assert panel.model_runtime_config["processing_resolution"] == 768
    assert panel.model_runtime_config["num_inference_steps"] == 10
    assert panel.model_runtime_config["ensemble_size"] == 3
    assert panel.model_runtime_config["output_uncertainty"] is True
    assert panel.model_runtime_config["model_cpu_offload"] is False
