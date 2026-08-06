from pathlib import Path
import numpy as np

from app.measurement_inspection_controller import MeasurementInspectionController


def test_controller_recompute_serializes_reset_flag(qtbot, tmp_path: Path, monkeypatch):
    state = tmp_path / "state.pkl"
    state.write_bytes(b"state")
    controller = MeasurementInspectionController(Path.cwd())
    captured = {}
    monkeypatch.setattr(controller, "_start_process", lambda args, state_value: captured.update(args=args, state=state_value))
    controller.recompute(
        base_state_path=state,
        observation_id="object-0",
        manual_mask=None,
        manual_revision=2,
        box_method="auto",
        reset_manual=True,
    )
    assert "--reset-manual" in captured["args"]
    assert "--manual-mask" not in captured["args"]


def test_controller_recompute_writes_dense_manual_mask(qtbot, tmp_path: Path, monkeypatch):
    state = tmp_path / "state.pkl"
    state.write_bytes(b"state")
    controller = MeasurementInspectionController(Path.cwd())
    captured = {}
    monkeypatch.setattr(controller, "_start_process", lambda args, state_value: captured.update(args=args, state=state_value))
    controller.recompute(
        base_state_path=state,
        observation_id="object-0",
        manual_mask=np.ones((8, 9), dtype=bool),
        manual_revision=4,
        box_method="oriented",
    )
    args = captured["args"]
    mask_path = Path(args[args.index("--manual-mask") + 1])
    assert np.load(mask_path, allow_pickle=False).dtype == np.bool_
    assert args[args.index("--manual-revision") + 1] == "4"
