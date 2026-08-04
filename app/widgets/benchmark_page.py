"""Interactive benchmark page built on the Phase 6 benchmark engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PyQt6.QtCore import Qt, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.benchmark_controller import (
    DEFAULT_OUTPUT_ROOT,
    BenchmarkRunConfiguration,
    BenchmarkRunState,
)
from app.theme import PAGE_MARGIN_BOTTOM, PAGE_MARGIN_H, PAGE_MARGIN_TOP, PAGE_SPACING
from core.hardware.gpu_info import detect_hardware
from core.hardware.profiles import ComputeProfile
from core.models.registry import check_model_availability, iter_model_descriptors

_MODEL_ID_ROLE = int(Qt.ItemDataRole.UserRole)
_DEVELOPMENT_ROLE = _MODEL_ID_ROLE + 1
_AVAILABILITY_ROLE = _MODEL_ID_ROLE + 2


class BenchmarkPage(QWidget):
    """Configure, run, cancel, inspect, and export model benchmarks."""

    run_requested = pyqtSignal(object)
    cancel_requested = pyqtSignal()

    _RESULT_COLUMNS = (
        ("Model", "display_name"),
        ("Status", "status"),
        ("Class", "classification"),
        ("Profile", "profile"),
        ("Load", "load_ms"),
        ("Warm-up", "warmup_ms"),
        ("Latency", "latency_ms"),
        ("FPS", "fps"),
        ("Model VRAM", "vram_gib"),
        ("Peak VRAM", "peak_vram_gib"),
        ("Output", "resolution"),
        ("Points", "point_count"),
        ("Scale", "depth_scale"),
        ("Intrinsics", "intrinsics_source"),
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("benchmarkPage")
        self._runtime_safe = True
        self._current_state = BenchmarkRunState.IDLE
        self._runtime_block_reason = ""
        self._last_output_dir: Path | None = None
        self._total_models = 0
        self._completed_models = 0
        self._build_ui()
        self._populate_models()
        self.apply_state(BenchmarkRunState.IDLE)

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(
            PAGE_MARGIN_H,
            PAGE_MARGIN_TOP,
            PAGE_MARGIN_H,
            PAGE_MARGIN_BOTTOM,
        )
        outer.setSpacing(PAGE_SPACING)

        outer.addWidget(self._build_title_card())

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_configuration_card())
        splitter.addWidget(self._build_results_card())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([255, 450])
        outer.addWidget(splitter, stretch=1)

    def _build_title_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("benchmarkCard")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(16, 11, 16, 11)
        layout.setSpacing(12)

        text = QVBoxLayout()
        text.setSpacing(2)
        title = QLabel("Interactive Cross-Model Benchmark")
        title.setObjectName("benchmarkTitle")
        subtitle = QLabel(
            "Registry-driven isolated model runs. Performance results do not claim ground-truth depth accuracy."
        )
        subtitle.setObjectName("benchmarkSubtitle")
        subtitle.setWordWrap(True)
        text.addWidget(title)
        text.addWidget(subtitle)
        layout.addLayout(text, stretch=1)

        self._status_badge = QLabel("READY")
        self._status_badge.setObjectName("benchmarkStatusBadge")
        self._status_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status_badge)
        return card

    def _build_configuration_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("benchmarkCard")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(14)

        model_column = QVBoxLayout()
        model_column.setSpacing(6)
        model_title_row = QHBoxLayout()
        title = QLabel("MODELS")
        title.setObjectName("benchmarkSectionTitle")
        model_title_row.addWidget(title)
        model_title_row.addStretch(1)
        self._select_all_btn = QPushButton("Select all")
        self._select_all_btn.setObjectName("benchmarkLinkButton")
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setObjectName("benchmarkLinkButton")
        self._select_all_btn.clicked.connect(self._select_all_production)
        self._clear_btn.clicked.connect(self._clear_selection)
        model_title_row.addWidget(self._select_all_btn)
        model_title_row.addWidget(self._clear_btn)
        model_column.addLayout(model_title_row)

        self._model_list = QListWidget()
        self._model_list.setObjectName("benchmarkModelList")
        self._model_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        model_column.addWidget(self._model_list, stretch=1)
        layout.addLayout(model_column, stretch=4)

        options = QGridLayout()
        options.setHorizontalSpacing(10)
        options.setVerticalSpacing(8)
        options.setColumnStretch(1, 1)

        self._profile_combo = QComboBox()
        for profile in ComputeProfile:
            self._profile_combo.addItem(profile.value.replace("_", " ").title(), profile)
        self._point_budget = QSpinBox()
        self._point_budget.setRange(1_000, 1_000_000)
        self._point_budget.setSingleStep(25_000)
        self._point_budget.setValue(100_000)
        self._point_budget.setGroupSeparatorShown(True)
        self._iterations = QSpinBox()
        self._iterations.setRange(1, 20)
        self._iterations.setValue(3)
        self._sequence_frames = QSpinBox()
        self._sequence_frames.setRange(2, 120)
        self._sequence_frames.setValue(12)
        self._image_asset = QComboBox()
        self._image_asset.addItem("Registry standard", "registry")
        self._image_asset.addItem("Landscape only", "landscape")
        self._image_asset.addItem("Portrait only", "portrait")
        self._sequence_asset = QComboBox()
        self._sequence_asset.addItem("Registry standard", "registry")
        self._sequence_asset.addItem("Temporal sequence", "temporal")
        self._require_all = QCheckBox("Fail run when any selected model is unavailable")
        self._smoke_test = QCheckBox("Smoke test (one image / two sequence frames)")

        row = 0
        row = self._add_option(options, row, "GPU profile", self._profile_combo)
        row = self._add_option(options, row, "Point budget", self._point_budget)
        row = self._add_option(options, row, "Image iterations", self._iterations)
        row = self._add_option(options, row, "Sequence frames", self._sequence_frames)
        row = self._add_option(options, row, "Image input", self._image_asset)
        row = self._add_option(options, row, "Sequence input", self._sequence_asset)
        options.addWidget(self._require_all, row, 0, 1, 2)
        row += 1
        options.addWidget(self._smoke_test, row, 0, 1, 2)
        layout.addLayout(options, stretch=5)

        action_column = QVBoxLayout()
        action_column.setSpacing(8)
        output_label = QLabel("OUTPUT DIRECTORY")
        output_label.setObjectName("benchmarkSectionTitle")
        action_column.addWidget(output_label)
        output_row = QHBoxLayout()
        self._output_path = QLineEdit(str(DEFAULT_OUTPUT_ROOT))
        self._output_path.setObjectName("benchmarkOutputPath")
        self._browse_output_btn = QPushButton("Browse")
        self._browse_output_btn.setObjectName("benchmarkSecondaryButton")
        self._browse_output_btn.clicked.connect(self._browse_output)
        output_row.addWidget(self._output_path, stretch=1)
        output_row.addWidget(self._browse_output_btn)
        action_column.addLayout(output_row)

        self._safety_message = QLabel()
        self._safety_message.setObjectName("benchmarkSafetyMessage")
        self._safety_message.setWordWrap(True)
        self._safety_message.hide()
        action_column.addWidget(self._safety_message)

        self._progress_label = QLabel("Ready to benchmark")
        self._progress_label.setObjectName("benchmarkProgressLabel")
        self._progress = QProgressBar()
        self._progress.setObjectName("benchmarkProgress")
        self._progress.setRange(0, 1)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        action_column.addStretch(1)
        action_column.addWidget(self._progress_label)
        action_column.addWidget(self._progress)

        buttons = QHBoxLayout()
        self._run_btn = QPushButton("Run Benchmark")
        self._run_btn.setObjectName("benchmarkPrimaryButton")
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setObjectName("benchmarkDangerButton")
        self._open_output_btn = QPushButton("Open Results")
        self._open_output_btn.setObjectName("benchmarkSecondaryButton")
        self._run_btn.clicked.connect(self._emit_run)
        self._cancel_btn.clicked.connect(self.cancel_requested.emit)
        self._open_output_btn.clicked.connect(self._open_output)
        buttons.addWidget(self._run_btn)
        buttons.addWidget(self._cancel_btn)
        buttons.addWidget(self._open_output_btn)
        action_column.addLayout(buttons)
        layout.addLayout(action_column, stretch=5)
        return card

    def _build_results_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("benchmarkCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel("RESULTS")
        title.setObjectName("benchmarkSectionTitle")
        header.addWidget(title)
        header.addStretch(1)
        self._summary_label = QLabel("No benchmark has been run in this session.")
        self._summary_label.setObjectName("benchmarkSubtitle")
        header.addWidget(self._summary_label)
        layout.addLayout(header)

        self._results = QTableWidget(0, len(self._RESULT_COLUMNS))
        self._results.setObjectName("benchmarkResultsTable")
        self._results.setHorizontalHeaderLabels([item[0] for item in self._RESULT_COLUMNS])
        self._results.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._results.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._results.setAlternatingRowColors(True)
        self._results.verticalHeader().setVisible(False)
        table_header = self._results.horizontalHeader()
        table_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table_header.setStretchLastSection(True)
        layout.addWidget(self._results, stretch=3)

        log_header = QHBoxLayout()
        log_title = QLabel("RUN LOG")
        log_title.setObjectName("benchmarkSectionTitle")
        log_header.addWidget(log_title)
        log_header.addStretch(1)
        clear_log = QPushButton("Clear log")
        clear_log.setObjectName("benchmarkLinkButton")
        clear_log.clicked.connect(lambda: self._log.clear())
        log_header.addWidget(clear_log)
        layout.addLayout(log_header)
        self._log = QTextEdit()
        self._log.setObjectName("benchmarkLog")
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(115)
        layout.addWidget(self._log, stretch=1)
        return card

    @staticmethod
    def _add_option(layout: QGridLayout, row: int, text: str, widget: QWidget) -> int:
        label = QLabel(text)
        label.setObjectName("benchmarkOptionLabel")
        widget.setObjectName(widget.objectName() or "benchmarkInput")
        layout.addWidget(label, row, 0)
        layout.addWidget(widget, row, 1)
        return row + 1

    def _populate_models(self) -> None:
        hardware = detect_hardware()
        first_production: QListWidgetItem | None = None
        for descriptor in iter_model_descriptors(selectable_only=True, enabled_only=True):
            availability = check_model_availability(descriptor.model_id, hardware)
            suffixes: list[str] = []
            if descriptor.development_only:
                suffixes.append("DEV ONLY")
            if not availability.available:
                suffixes.append("UNAVAILABLE")
            suffix = f"  ·  {' / '.join(suffixes)}" if suffixes else ""
            item = QListWidgetItem(f"{descriptor.display_name}{suffix}")
            item.setData(_MODEL_ID_ROLE, descriptor.model_id)
            item.setData(_DEVELOPMENT_ROLE, descriptor.development_only)
            item.setData(_AVAILABILITY_ROLE, availability.available)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Unchecked)
            tooltip = descriptor.presentation.description
            if availability.reason:
                tooltip += f"\n\nAvailability: {availability.reason}"
            item.setToolTip(tooltip)
            self._model_list.addItem(item)
            if first_production is None and not descriptor.development_only:
                first_production = item
        if first_production is not None:
            first_production.setCheckState(Qt.CheckState.Checked)

    def selected_model_ids(self) -> tuple[str, ...]:
        selected: list[str] = []
        for index in range(self._model_list.count()):
            item = self._model_list.item(index)
            if item.checkState() == Qt.CheckState.Checked:
                selected.append(str(item.data(_MODEL_ID_ROLE)))
        return tuple(selected)

    def _select_all_production(self) -> None:
        for index in range(self._model_list.count()):
            item = self._model_list.item(index)
            development_only = bool(item.data(_DEVELOPMENT_ROLE))
            item.setCheckState(
                Qt.CheckState.Unchecked if development_only else Qt.CheckState.Checked
            )

    def _clear_selection(self) -> None:
        for index in range(self._model_list.count()):
            self._model_list.item(index).setCheckState(Qt.CheckState.Unchecked)

    def _browse_output(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select Benchmark Output Directory",
            self._output_path.text().strip() or str(DEFAULT_OUTPUT_ROOT),
        )
        if selected:
            self._output_path.setText(selected)

    def _emit_run(self) -> None:
        if not self._runtime_safe:
            self._show_safety_message(self._runtime_block_reason)
            return
        model_ids = self.selected_model_ids()
        if not model_ids:
            self._show_safety_message("Select at least one model before starting.")
            return
        output_text = self._output_path.text().strip()
        output = Path(output_text).expanduser() if output_text else None
        configuration = BenchmarkRunConfiguration(
            model_ids=model_ids,
            profile=self._profile_combo.currentData(),
            point_budget=self._point_budget.value(),
            iterations=self._iterations.value(),
            sequence_frames=self._sequence_frames.value(),
            image_asset=str(self._image_asset.currentData()),
            sequence_asset=str(self._sequence_asset.currentData()),
            require_all=self._require_all.isChecked(),
            smoke_test=self._smoke_test.isChecked(),
            output_dir=output,
        )
        self._safety_message.hide()
        self._results.setRowCount(0)
        self._summary_label.setText("Benchmark starting…")
        self.run_requested.emit(configuration)

    def set_runtime_safe(self, safe: bool, reason: str = "") -> None:
        self._runtime_safe = bool(safe)
        self._runtime_block_reason = str(reason)
        if safe:
            self._safety_message.hide()
        else:
            self._show_safety_message(reason)
        self.apply_state(self._current_state)

    def _show_safety_message(self, message: str) -> None:
        self._safety_message.setText(message)
        self._safety_message.setVisible(bool(message))

    def apply_state(self, state: BenchmarkRunState) -> None:
        self._current_state = state
        running = state in {
            BenchmarkRunState.STARTING,
            BenchmarkRunState.RUNNING,
            BenchmarkRunState.CANCELLING,
        }
        controls_enabled = not running
        for widget in (
            self._model_list,
            self._select_all_btn,
            self._clear_btn,
            self._profile_combo,
            self._point_budget,
            self._iterations,
            self._sequence_frames,
            self._image_asset,
            self._sequence_asset,
            self._require_all,
            self._smoke_test,
            self._output_path,
            self._browse_output_btn,
        ):
            widget.setEnabled(controls_enabled)
        self._run_btn.setEnabled(controls_enabled and self._runtime_safe)
        self._cancel_btn.setEnabled(running and state is not BenchmarkRunState.CANCELLING)
        self._open_output_btn.setEnabled(self._last_output_dir is not None)

        labels = {
            BenchmarkRunState.IDLE: ("READY", "ready"),
            BenchmarkRunState.STARTING: ("STARTING", "running"),
            BenchmarkRunState.RUNNING: ("RUNNING", "running"),
            BenchmarkRunState.CANCELLING: ("CANCELLING", "warning"),
            BenchmarkRunState.COMPLETED: ("COMPLETED", "ready"),
            BenchmarkRunState.COMPLETED_WITH_ERRORS: ("COMPLETED WITH ERRORS", "warning"),
            BenchmarkRunState.CANCELLED: ("CANCELLED", "warning"),
            BenchmarkRunState.FAILED: ("FAILED", "error"),
        }
        text, kind = labels[state]
        self._status_badge.setText(text)
        self._status_badge.setProperty("statusKind", kind)
        self._status_badge.style().unpolish(self._status_badge)
        self._status_badge.style().polish(self._status_badge)
        if state is BenchmarkRunState.IDLE:
            self._progress_label.setText("Ready to benchmark")
        elif state is BenchmarkRunState.CANCELLING:
            self._progress_label.setText("Stopping the active isolated model…")

    def handle_event(self, event: dict[str, Any]) -> None:
        event_name = str(event.get("event", ""))
        if event_name == "benchmark_started":
            self._total_models = int(event.get("total_models", 0))
            self._completed_models = 0
            self._progress.setRange(0, max(1, self._total_models))
            self._progress.setValue(0)
            self._progress_label.setText(
                f"Preparing {self._total_models} isolated model run(s)…"
            )
        elif event_name == "model_started":
            index = int(event.get("index", 1))
            total = int(event.get("total_models", self._total_models or 1))
            name = str(event.get("display_name", event.get("model_id", "model")))
            self._progress_label.setText(f"{index}/{total} · Running {name}")
        elif event_name == "model_completed":
            self._completed_models = int(event.get("index", self._completed_models + 1))
            self._progress.setValue(self._completed_models)
            name = str(event.get("display_name", event.get("model_id", "model")))
            status = str(event.get("status", "unknown")).upper()
            self.append_log(f"{name}: {status}")
        elif event_name == "cancellation_requested":
            self._progress_label.setText("Cancellation requested…")
        elif event_name == "benchmark_finished":
            completed = int(event.get("completed_models", self._completed_models))
            total = int(event.get("total_models", self._total_models or completed))
            self._progress.setRange(0, max(1, total))
            self._progress.setValue(completed)
            if event.get("cancelled"):
                self._progress_label.setText(f"Cancelled after {completed}/{total} model(s)")
            else:
                self._progress_label.setText(f"Completed {completed}/{total} model(s)")

    def append_log(self, text: str) -> None:
        if text:
            self._log.append(str(text))

    def set_results(self, payload: dict[str, Any], output_dir: str) -> None:
        self._last_output_dir = Path(output_dir)
        self._open_output_btn.setEnabled(True)
        results = list(payload.get("results", []))
        self._results.setRowCount(len(results))
        for row, result in enumerate(results):
            flattened = self._flatten_result(result)
            for column, (_title, key) in enumerate(self._RESULT_COLUMNS):
                item = QTableWidgetItem(str(flattened.get(key, "—")))
                if key in {
                    "load_ms",
                    "warmup_ms",
                    "latency_ms",
                    "fps",
                    "vram_gib",
                    "peak_vram_gib",
                    "point_count",
                }:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._results.setItem(row, column, item)
        passed = sum(1 for item in results if item.get("status") == "passed")
        unavailable = sum(1 for item in results if item.get("status") == "unavailable")
        failed = len(results) - passed - unavailable
        status = "cancelled" if payload.get("cancelled") else "completed"
        self._summary_label.setText(
            f"{status.title()} · {passed} passed · {failed} failed · {unavailable} unavailable"
        )

    @staticmethod
    def _flatten_result(result: dict[str, Any]) -> dict[str, str]:
        timing = result.get("timing", {}) or {}
        memory = result.get("memory", {}) or {}
        output = result.get("output", {}) or {}
        runtime = result.get("runtime", {}) or {}
        resolution = output.get("resolution") or []
        return {
            "display_name": str(result.get("display_name") or result.get("model_id") or "—"),
            "status": str(result.get("status") or "—").upper(),
            "classification": str(result.get("classification") or "—"),
            "profile": str(runtime.get("effective_profile") or "—"),
            "load_ms": BenchmarkPage._format_ms(timing.get("load_ms")),
            "warmup_ms": BenchmarkPage._format_ms(timing.get("warmup_ms")),
            "latency_ms": BenchmarkPage._format_ms(
                (timing.get("model_latency_ms") or {}).get("mean")
            ),
            "fps": BenchmarkPage._format_number(timing.get("fps")),
            "vram_gib": BenchmarkPage._format_gib(memory.get("reserved_gib")),
            "peak_vram_gib": BenchmarkPage._format_gib(memory.get("peak_reserved_gib")),
            "resolution": "×".join(str(value) for value in resolution) if resolution else "—",
            "point_count": (
                f"{int(output.get('point_count')):,}"
                if output.get("point_count") is not None
                else "—"
            ),
            "depth_scale": str(output.get("depth_scale") or "—"),
            "intrinsics_source": str(output.get("intrinsics_source") or "—"),
        }

    @staticmethod
    def _format_ms(value: object) -> str:
        return "—" if value is None else f"{float(value):.1f} ms"

    @staticmethod
    def _format_gib(value: object) -> str:
        return "—" if value is None else f"{float(value):.2f} GiB"

    @staticmethod
    def _format_number(value: object) -> str:
        return "—" if value is None else f"{float(value):.2f}"

    def _open_output(self) -> None:
        if self._last_output_dir is None:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_output_dir)))
