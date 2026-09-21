from pathlib import Path


def test_controller_has_fail_closed_i6_2c_result_handoff() -> None:
    source = Path("app/reconstruction_controller.py").read_text(encoding="utf-8")

    assert "from app.reconstruction_offline_result_handoff import" in source
    assert "offline_result_handoff_ready = pyqtSignal(object)" in source
    assert "def last_offline_result_handoff(" in source
    assert "build_offline_result_handoff(" in source
    assert "self._persistence_controller.accept_offline_result(payload, output_dir)" in source
    assert "self.offline_result_handoff_ready.emit(handoff)" in source
    assert "self.offline_result_ready.emit(payload, output_dir)" in source
    assert "Offline reconstruction result handoff rejected result" in source

    handler = source[source.index("    def _on_offline_result_ready(") :]
    build_at = handler.index("handoff = build_offline_result_handoff(")
    persist_at = handler.index("self._persistence_controller.accept_offline_result(payload, output_dir)")
    handoff_emit_at = handler.index("self.offline_result_handoff_ready.emit(handoff)")
    legacy_emit_at = handler.index("self.offline_result_ready.emit(payload, output_dir)")
    assert build_at < persist_at < handoff_emit_at < legacy_emit_at


def test_new_start_clears_only_passive_terminal_handoff_cache() -> None:
    source = Path("app/reconstruction_controller.py").read_text(encoding="utf-8")
    assert "self._last_offline_result_handoff = None" in source
    assert "if state is OfflineReconstructionRunState.STARTING:" in source
