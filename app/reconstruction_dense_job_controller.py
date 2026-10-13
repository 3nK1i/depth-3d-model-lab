"""Qt-owned explicit post-sparse dense job; process-tree cancellation and child verification.

I6.7E production bridge. Full reconstruction/PLY scans run only in isolated child
processes. An I6.7C preview has no effect on the final-admission decision.
"""
from __future__ import annotations

from enum import Enum
import json
import os
from pathlib import Path
import shutil
import signal
import sys
from uuid import uuid4

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, pyqtSignal

from app.reconstruction_job_telemetry_relay import JobTelemetryQtRelay
from app.reconstruction_offline_result_handoff import (
    OfflineReconstructionResultHandoff, OfflineResultHandoffAvailability,
)
from core.reconstruction.errors import ReconstructionValidationError
from core.reconstruction.mvs_provider import I43B_PROVIDER_ID
from core.reconstruction.reconstruction_progressive_dense_transport import (
    I67C_MAX_TRANSPORT_SNAPSHOT_BYTES, PROGRESSIVE_DENSE_EVENT_PREFIX,
    parse_progressive_dense_transport_line,
)

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / 'scripts/run_i6_7e2b2a_post_sparse_dense_worker.py'
VERIFY_WORKER = ROOT / 'scripts/run_i6_7e_dense_completion_verify.py'
WORKER_PROTOCOL = 'depth3d.reconstruction.dense.post_sparse.v1'
VERIFY_PROTOCOL = 'depth3d.reconstruction.dense.verify.v1'
MAX_LINE_BYTES = I67C_MAX_TRANSPORT_SNAPSHOT_BYTES + 1


class DenseJobState(str, Enum):
    IDLE = 'idle'
    STARTING = 'starting'
    RUNNING = 'running'
    VERIFYING = 'verifying'
    CANCELLING = 'cancelling'
    COMPLETED = 'completed'
    CANCELLED = 'cancelled'
    FAILED = 'failed'


class ExplicitDenseJobController(QObject):
    """One admitted sparse session -> one explicit dense run -> independently verified artifact."""
    state_changed = pyqtSignal(object)
    provisional_snapshot_received = pyqtSignal(object)
    telemetry_received = pyqtSignal(object)  # passive; never completion or admission
    log_received = pyqtSignal(str)
    admission_ready = pyqtSignal(object)  # E2B1-verified compact dict, *not* geometry
    run_finished = pyqtSignal(object)
    error_occurred = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None, *, worker_script: Path = WORKER,
                 verifier_script: Path = VERIFY_WORKER) -> None:
        super().__init__(parent)
        self._worker_script = Path(worker_script).resolve()
        self._verifier_script = Path(verifier_script).resolve()
        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self._process.readyReadStandardOutput.connect(self._read_output)
        self._process.finished.connect(self._finished)
        self._process.errorOccurred.connect(self._process_error)
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._force_kill_group)
        self._telemetry_relay = JobTelemetryQtRelay(self)
        self._telemetry_relay.observation_received.connect(self.telemetry_received.emit)
        self._state = DenseJobState.IDLE
        self._phase = ''
        self._line_buffer = b''
        self._pgid: int | None = None
        self._session_id = ''
        self._dense_run_id = ''
        self._output_dir: Path | None = None
        self._claimed_admission: dict | None = None
        self._verified_admission: dict | None = None
        self._cancel_requested = False
        self._protocol_failed = False
        self._shutdown = False
        self._finalized = True
        self._error = ''

    @property
    def state(self) -> DenseJobState:
        return self._state

    @property
    def is_active(self) -> bool:
        return not self._finalized or self._process.state() != QProcess.ProcessState.NotRunning

    @property
    def dense_run_id(self) -> str:
        return self._dense_run_id

    @property
    def sparse_session_id(self) -> str:
        return self._session_id

    def _state_to(self, state: DenseJobState) -> None:
        if state != self._state:
            self._state = state
            self.state_changed.emit(state)

    def start(self, *, admitted_sparse: OfflineReconstructionResultHandoff,
              pycolmap_python: Path, provider_id: str = I43B_PROVIDER_ID,
              timeout_minutes: float = 120.0) -> str:
        if self._shutdown or self.is_active:
            raise ReconstructionValidationError('Explicit dense job is shut down or another run is active.')
        if provider_id != I43B_PROVIDER_ID:
            raise ReconstructionValidationError('Explicit COLMAP-MVS is required; production Auto is forbidden.')
        if not isinstance(admitted_sparse, OfflineReconstructionResultHandoff) or (
            admitted_sparse.availability is not OfflineResultHandoffAvailability.PARTIAL
            or admitted_sparse.status != 'completed' or not admitted_sparse.camera_pose_available
            or admitted_sparse.canonical_geometry_authoritative or admitted_sparse.persistent_scene_ready
            or admitted_sparse.scale_status != 'arbitrary'
        ):
            raise ReconstructionValidationError('Only a completed, admitted arbitrary-scale sparse result may start MVS.')
        if self._state is not DenseJobState.IDLE and admitted_sparse.session_id == self._session_id:
            raise ReconstructionValidationError(
                'Dense reconstruction was already attempted for this sparse session; '
                'preserve its completion or failure evidence instead of retrying automatically.')
        root = Path(admitted_sparse.output_dir)
        if not root.is_absolute() or not root.is_dir() or root.is_symlink() or (
            Path(admitted_sparse.result_path) != root / 'offline_result.json'
        ):
            raise ReconstructionValidationError('Sparse result output path or identity is invalid.')
        py = Path(pycolmap_python).expanduser().resolve(strict=True)
        if not py.is_file() or not self._worker_script.is_file() or not self._verifier_script.is_file():
            raise ReconstructionValidationError('Required Python or dense worker script is missing.')
        if not 0 < float(timeout_minutes) <= 720:
            raise ReconstructionValidationError('Dense worker timeout must be bounded.')
        run_id = str(uuid4())
        if (root / 'dense_runs' / run_id).exists():
            raise ReconstructionValidationError('Dense run already exists; refusing overwrite.')
        self._session_id = admitted_sparse.session_id
        self._dense_run_id = run_id
        self._output_dir = root
        self._claimed_admission = None
        self._verified_admission = None
        self._cancel_requested = False
        self._protocol_failed = False
        self._error = ''
        self._finalized = False
        self._state_to(DenseJobState.STARTING)
        arguments = [str(self._worker_script), '--enable-explicit-colmap-mvs',
                     '--provider-id', I43B_PROVIDER_ID, '--offline-output-dir', str(root),
                     '--session-id', self._session_id, '--pycolmap-python', str(py),
                     '--dense-run-id', run_id, '--timeout-minutes', str(float(timeout_minutes))]
        try:
            self._launch_process('worker', arguments)
        except Exception:
            self._finalized = True
            self._state_to(DenseJobState.FAILED)
            raise
        self._telemetry_relay.begin('post_sparse_dense', self._session_id,
                                    run_id, root / 'dense_runs' / run_id)
        self._state_to(DenseJobState.RUNNING)
        return run_id

    def _launch_process(self, phase: str, script_args: list[str]) -> None:
        # On Ubuntu, setsid creates a *new session/process group*, inherited by
        # all COLMAP/PyCOLMAP children. Killing the QProcess parent alone is unsafe.
        setsid = shutil.which('setsid')
        if os.name != 'posix' or not hasattr(os, 'killpg') or not setsid:
            raise ReconstructionValidationError('Dense job requires a Linux setsid process-group supervisor.')
        self._phase = phase
        self._line_buffer = b''
        self._pgid = None
        env = QProcessEnvironment.systemEnvironment()
        env.insert('PYTHONUNBUFFERED', '1')
        self._process.setProcessEnvironment(env)
        self._process.setWorkingDirectory(str(ROOT))
        self._process.setProgram(setsid)
        self._process.setArguments([sys.executable, *script_args])
        self._process.start()
        if not self._process.waitForStarted(5000):
            raise ReconstructionValidationError('Could not start supervised dense job: ' + self._process.errorString())
        pid = int(self._process.processId())
        # The child may not have called setsid at the instant waitForStarted
        # returns. Verify before allowing a GUI cancellation; never signal GUI PGID.
        if pid > 0:
            try:
                pgid = os.getpgid(pid)
                if pgid == pid and pgid != os.getpgrp():
                    self._pgid = pgid
            except (ProcessLookupError, PermissionError, OSError):
                pass

    def _ensure_group_identity(self) -> None:
        if self._pgid is not None:
            return
        pid = int(self._process.processId())
        if pid > 0:
            try:
                pgid = os.getpgid(pid)
                if pgid == pid and pgid != os.getpgrp():
                    self._pgid = pgid
            except (OSError, ProcessLookupError):
                pass

    def _signal_group(self, sig: int) -> None:
        self._ensure_group_identity()
        if self._pgid is not None and self._pgid != os.getpgrp():
            try:
                os.killpg(self._pgid, sig)
            except ProcessLookupError:
                pass
        elif self._process.state() != QProcess.ProcessState.NotRunning:
            # No verified group -> never killpg a guessed PID. Stop parent;
            # finalization remains a failure, not successful cancellation.
            self._protocol_failed = True
            self._error = 'Cannot establish isolated worker process group.'
            self._process.kill()

    def cancel(self) -> bool:
        if self._shutdown or not self.is_active or self._cancel_requested:
            return False
        self._cancel_requested = True
        self._state_to(DenseJobState.CANCELLING)
        if self._process.state() != QProcess.ProcessState.NotRunning:
            self._signal_group(signal.SIGTERM)
            self._kill_timer.start(8000)
        else:
            self._complete(DenseJobState.CANCELLED)
        return True

    def _force_kill_group(self) -> None:
        self._signal_group(signal.SIGKILL)

    def _read_output(self) -> None:
        incoming = bytes(self._process.readAllStandardOutput())
        if not incoming or self._finalized:
            return
        self._line_buffer += incoming
        if len(self._line_buffer) > MAX_LINE_BYTES + 65536:
            self._protocol_failure('Worker output exceeded bounded transport buffer.')
            return
        while b'\n' in self._line_buffer:
            line, self._line_buffer = self._line_buffer.split(b'\n', 1)
            if len(line) > MAX_LINE_BYTES:
                self._protocol_failure('Oversize worker line rejected.')
                return
            self._consume_line(line.decode('utf-8', errors='replace').rstrip('\r'))
            if self._protocol_failed or self._finalized:
                return

    def _consume_line(self, line: str) -> None:
        if not line or self._cancel_requested:
            return
        if self._phase == 'worker' and self._pgid is None and self._process.state() != QProcess.ProcessState.NotRunning:
            self._ensure_group_identity()
            if self._pgid is None:
                self._protocol_failure('Worker did not enter an isolated process group.')
                return
        if line.startswith(PROGRESSIVE_DENSE_EVENT_PREFIX) and self._phase == 'worker':
            try:
                snapshot = parse_progressive_dense_transport_line(line)
                if snapshot is None or (
                    snapshot.source_sfm_session_id != self._session_id or
                    snapshot.dense_run_id != self._dense_run_id or
                    snapshot.provider_id != I43B_PROVIDER_ID or
                    snapshot.reconstruction_frame_id != 'i6_7e2b2:pycolmap_sfm:' + self._session_id
                ):
                    raise ReconstructionValidationError('Provisional dense snapshot identity mismatch.')
            except (ValueError, TypeError, ReconstructionValidationError) as exc:
                # Preview is not admission; reject bad snapshot without accepting it.
                self.log_received.emit('Rejected dense preview: ' + str(exc)[:400])
                return
            self.provisional_snapshot_received.emit(snapshot)
            return
        try:
            payload = json.loads(line)
        except ValueError:
            self.log_received.emit(line[:500])
            return
        if not isinstance(payload, dict):
            return
        if self._phase == 'worker' and payload.get('protocol') == WORKER_PROTOCOL:
            if (payload.get('session_id') != self._session_id or
                payload.get('dense_run_id') != self._dense_run_id):
                self._protocol_failure('Foreign dense worker session/run identity.')
                return
            kind = payload.get('event')
            if kind == 'dense_completed':
                admission = payload.get('admission')
                if (self._claimed_admission is not None or not isinstance(admission, dict) or
                    payload.get('final_dense_result_admitted') is not True or
                    admission.get('sparse_session_id') != self._session_id or
                    admission.get('dense_run_id') != self._dense_run_id or
                    admission.get('provider_id') != I43B_PROVIDER_ID or
                    admission.get('final_dense_result_admitted') is not True):
                    self._protocol_failure('Invalid/duplicate dense completion claim.')
                    return
                self._claimed_admission = admission
            elif kind == 'dense_failed':
                self._error = str(payload.get('detail', 'Dense worker failed.'))[:800]
            elif kind == 'dense_starting':
                self.log_received.emit('Explicit COLMAP-MVS dense reconstruction started.')
            else:
                self._protocol_failure('Unknown dense worker control event.')
        elif self._phase == 'verifier' and payload.get('protocol') == VERIFY_PROTOCOL:
            if payload.get('event') == 'dense_verified':
                admission = payload.get('admission')
                if (self._verified_admission is not None or not isinstance(admission, dict) or
                    payload.get('session_id') != self._session_id or
                    payload.get('dense_run_id') != self._dense_run_id or
                    admission != self._claimed_admission):
                    self._protocol_failure('Independent dense verification returned foreign admission.')
                    return
                self._verified_admission = admission
            elif payload.get('event') == 'dense_verification_failed':
                self._error = str(payload.get('detail', 'Dense verification failed.'))[:800]
            else:
                self._protocol_failure('Unknown verification control event.')
        else:
            self.log_received.emit(line[:500])

    def _protocol_failure(self, detail: str) -> None:
        self._protocol_failed = True
        self._error = detail
        if self._process.state() != QProcess.ProcessState.NotRunning:
            self._signal_group(signal.SIGTERM)
            self._kill_timer.start(8000)
        else:
            self._complete(DenseJobState.FAILED)

    def _finished(self, code: int, status: QProcess.ExitStatus) -> None:
        if self._phase == 'worker':
            self._telemetry_relay.poll()
            self._telemetry_relay.close()  # verifier is a separate process, not this worker
        self._kill_timer.stop()
        # Qt can deliver finished() before the final readyRead notification.
        # Never discard the terminal dense_completed/verified line on that race.
        self._read_output()
        if self._finalized:
            return
        if self._line_buffer.strip():
            if len(self._line_buffer) <= MAX_LINE_BYTES:
                self._consume_line(self._line_buffer.decode('utf-8', errors='replace'))
            else:
                self._protocol_failed = True
                self._error = 'Truncated/oversize terminal protocol.'
        self._line_buffer = b''
        # The leader can exit before nested children: kill the isolated group
        # on cancellation/failure, never leave orphan PatchMatch/Fusion workers.
        failed = self._cancel_requested or self._protocol_failed or code != 0 or (
            status != QProcess.ExitStatus.NormalExit
        )
        if failed:
            self._signal_group(signal.SIGKILL)
            if self._cancel_requested:
                self._complete(DenseJobState.CANCELLED)
            else:
                self._complete(DenseJobState.FAILED)
            return
        if self._phase == 'worker':
            if self._claimed_admission is None:
                self._signal_group(signal.SIGKILL)
                self._error = 'Worker exited without a final-admission claim.'
                self._complete(DenseJobState.FAILED)
                return
            self._state_to(DenseJobState.VERIFYING)
            # Avoid launching a second QProcess synchronously inside finished().
            QTimer.singleShot(0, self._launch_verifier)
            return
        if self._phase == 'verifier' and self._verified_admission is not None:
            self.admission_ready.emit(self._verified_admission)
            self._complete(DenseJobState.COMPLETED)
            return
        self._error = self._error or 'Independent dense admission verification was not completed.'
        self._complete(DenseJobState.FAILED)

    def _launch_verifier(self) -> None:
        if self._cancel_requested or self._finalized:
            return
        claim = self._claimed_admission or {}
        try:
            args = [str(self._verifier_script), '--offline-output-dir', str(self._output_dir),
                    '--session-id', self._session_id, '--dense-run-id', self._dense_run_id,
                    '--manifest-sha256', str(claim['manifest_sha256']),
                    '--geometry-sha256', str(claim['complete_geometry_sha256'])]
            self._launch_process('verifier', args)
        except (OSError, KeyError, ReconstructionValidationError, ValueError) as exc:
            self._error = 'Could not start independent dense admission verifier: ' + str(exc)[:500]
            self._complete(DenseJobState.FAILED)

    def _process_error(self, _error: QProcess.ProcessError) -> None:
        if self.is_active and not self._cancel_requested:
            self.log_received.emit('Dense worker process error: ' + self._process.errorString()[:400])

    def _complete(self, state: DenseJobState) -> None:
        if self._finalized:
            return
        self._telemetry_relay.close()
        self._kill_timer.stop()
        self._finalized = True
        self._state_to(state)
        if state == DenseJobState.FAILED:
            self.error_occurred.emit(self._error or 'Dense worker exited without verified completion.')
        self.run_finished.emit(dict(state=state, session_id=self._session_id,
                                    dense_run_id=self._dense_run_id,
                                    final_dense_result_admitted=(state == DenseJobState.COMPLETED)))

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._telemetry_relay.close()
        if self.is_active:
            self.cancel()
            if self._process.state() != QProcess.ProcessState.NotRunning:
                if not self._process.waitForFinished(8000):
                    self._force_kill_group()
                    self._process.waitForFinished(2000)
        self._shutdown = True
