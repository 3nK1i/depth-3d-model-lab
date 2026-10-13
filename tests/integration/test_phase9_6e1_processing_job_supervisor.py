from __future__ import annotations

import os
from pathlib import Path
import sys
import textwrap
from uuid import uuid4

import pytest

pytest.importorskip("PyQt6")

from app.reconstruction_processing_job_controller import DigitalTwinProcessingJobController
from app.reconstruction_processing_jobs import (
    DigitalTwinProcessingCommand,
    DigitalTwinProcessingJobSpec,
    DigitalTwinProcessingKind,
    DigitalTwinProcessingLifecycle,
)


PROTOCOL = "depth3d.digital-twin.processing.test.v1"


def _worker(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            f'''
            import json, os, signal, subprocess, sys, time
            from pathlib import Path

            mode = sys.argv[1]
            marker = Path(sys.argv[2])
            protocol = {PROTOCOL!r}
            job_id = os.environ["DEPTH3D_DIGITAL_TWIN_JOB_ID"]

            def emit(event, **payload):
                print(json.dumps({{"protocol": protocol, "job_id": job_id, "event": event, **payload}}), flush=True)

            if mode == "complete":
                emit("started", stage="prepare", message="ready", progress_fraction=0.0)
                emit("progress", stage="optimization", message="working", progress_fraction=0.5,
                     completed_units=5, total_units=10)
                marker.write_text("artifact", encoding="utf-8")
                emit("progress", stage="persist", message="done", progress_fraction=1.0,
                     completed_units=10, total_units=10, result_path=str(marker))
                raise SystemExit(0)

            if mode == "unsafe":
                emit("progress", stage="bad", message="unsafe", progress_fraction=0.1,
                     certified_metrology=True)
                time.sleep(60)

            if mode == "stubborn":
                child = subprocess.Popen([
                    sys.executable,
                    "-c",
                    "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)",
                ])
                marker.write_text(str(child.pid), encoding="utf-8")
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                emit("progress", stage="running", message="stubborn", progress_fraction=0.2)
                while True:
                    time.sleep(0.05)

            raise SystemExit(3)
            '''
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def _spec(tmp_path: Path, name: str) -> DigitalTwinProcessingJobSpec:
    return DigitalTwinProcessingJobSpec(
        job_id=str(uuid4()),
        kind=DigitalTwinProcessingKind.NERF_APPEARANCE,
        title="Build NeRF Appearance",
        backend_label="fixture-backend",
        runtime_label="external-fixture-runtime",
        output_dir=tmp_path / f"out-{name}",
        expected_protocol=PROTOCOL,
        scene_id="scene-e1",
        profile_label="Test",
    )


def _command(tmp_path: Path, worker: Path, mode: str, marker: Path) -> DigitalTwinProcessingCommand:
    return DigitalTwinProcessingCommand(
        program=Path(sys.executable),
        arguments=(str(worker), mode, str(marker)),
        working_directory=tmp_path,
    )


def test_e1_supervisor_reports_progress_and_completes_shell_free_job(tmp_path: Path, qtbot) -> None:
    worker = tmp_path / "worker.py"
    spec = _spec(tmp_path, "complete")
    marker = spec.output_dir / "result.bin"
    _worker(worker)
    controller = DigitalTwinProcessingJobController()
    events: list[dict] = []
    controller.event_received.connect(events.append)
    try:
        controller.start(spec, _command(tmp_path, worker, "complete", marker))
        qtbot.waitUntil(
            lambda: controller.snapshot.lifecycle is DigitalTwinProcessingLifecycle.COMPLETED,
            timeout=4_000,
        )
        assert controller.is_running is False
        assert controller.snapshot.progress_fraction == pytest.approx(1.0)
        assert controller.snapshot.completed_units == 10
        assert controller.snapshot.total_units == 10
        assert controller.snapshot.result_path == str(marker)
        assert marker.is_file()
        assert any(item.get("stage") == "optimization" for item in events)
        assert controller.snapshot.spec is not None
        assert "canonical authority unchanged" in controller.snapshot.authority_text
    finally:
        controller.shutdown()


def test_e1_supervisor_rejects_backend_authority_promotion(tmp_path: Path, qtbot) -> None:
    worker = tmp_path / "worker.py"
    marker = tmp_path / "unsafe.bin"
    _worker(worker)
    controller = DigitalTwinProcessingJobController()
    controller._kill_timer.setInterval(150)
    errors: list[str] = []
    controller.error_occurred.connect(errors.append)
    try:
        controller.start(_spec(tmp_path, "unsafe"), _command(tmp_path, worker, "unsafe", marker))
        qtbot.waitUntil(
            lambda: controller.snapshot.lifecycle is DigitalTwinProcessingLifecycle.FAILED,
            timeout=4_000,
        )
        assert controller.is_running is False
        assert any("certified metrology" in item for item in errors)
        assert "unsafe certified metrology claim" in controller.snapshot.error
    finally:
        controller.shutdown()


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="process-group hardening is POSIX-only")
def test_e1_cancel_kills_external_process_group_without_orphan(tmp_path: Path, qtbot) -> None:
    worker = tmp_path / "worker.py"
    marker = tmp_path / "child_pid.txt"
    _worker(worker)
    controller = DigitalTwinProcessingJobController()
    controller._kill_timer.setInterval(150)
    try:
        controller.start(_spec(tmp_path, "stubborn"), _command(tmp_path, worker, "stubborn", marker))
        qtbot.waitUntil(marker.is_file, timeout=2_000)
        child_pid = int(marker.read_text(encoding="utf-8"))
        assert child_pid > 1
        assert controller.cancel() is True
        qtbot.waitUntil(
            lambda: controller.snapshot.lifecycle is DigitalTwinProcessingLifecycle.CANCELLED,
            timeout=4_000,
        )
        assert controller.is_running is False

        def child_gone_or_zombie() -> bool:
            stat = Path(f"/proc/{{child_pid}}/stat")
            if not stat.exists():
                return True
            try:
                fields = stat.read_text(encoding="utf-8").split()
            except OSError:
                return True
            return len(fields) > 2 and fields[2] == "Z"

        qtbot.waitUntil(child_gone_or_zombie, timeout=2_000)
    finally:
        controller.shutdown()
