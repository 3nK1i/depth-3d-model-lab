"""Process-safe application logging and uncaught-exception diagnostics.

The Qt process and inference worker write to separate rotating log files so a
worker traceback never disappears into a multiprocessing stderr stream and two
processes never rotate the same file concurrently.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import tempfile
import threading
from pathlib import Path
from types import TracebackType
from typing import Callable

_DEFAULT_MAX_BYTES = 5_000_000
_DEFAULT_BACKUP_COUNT = 5
_MANAGED_HANDLER_ATTRIBUTE = "_depth3d_managed_handler"


def project_root() -> Path:
    """Return the repository/application root from the installed source tree."""

    return Path(__file__).resolve().parents[2]


def configure_logging(
    *,
    process_name: str = "app",
    log_dir: str | os.PathLike[str] | None = None,
    level: str | int | None = None,
    console: bool | None = None,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    backup_count: int = _DEFAULT_BACKUP_COUNT,
) -> Path:
    """Configure idempotent rotating logs and return the active log path.

    ``DEPTH3D_LOG_DIR`` and ``DEPTH3D_LOG_LEVEL`` provide deployment-time
    overrides without modifying source files. The worker defaults to file-only
    logging to avoid duplicating every message in the parent terminal.
    """

    normalized_name = _safe_process_name(process_name)
    requested_directory = Path(
        log_dir
        if log_dir is not None
        else os.environ.get("DEPTH3D_LOG_DIR", project_root() / "logs")
    ).expanduser()
    directory = _select_log_directory(requested_directory)
    log_path = directory / f"{normalized_name}.log"
    resolved_level = _resolve_level(level or os.environ.get("DEPTH3D_LOG_LEVEL", "INFO"))
    if console is None:
        console = normalized_name == "app"

    root_logger = logging.getLogger()
    root_logger.setLevel(resolved_level)
    _remove_managed_handlers(root_logger)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(processName)s:%(process)d | "
        "%(threadName)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=max(1, int(max_bytes)),
        backupCount=max(0, int(backup_count)),
        encoding="utf-8",
        delay=True,
    )
    file_handler.setLevel(resolved_level)
    file_handler.setFormatter(formatter)
    setattr(file_handler, _MANAGED_HANDLER_ATTRIBUTE, True)
    root_logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(resolved_level)
        console_handler.setFormatter(formatter)
        setattr(console_handler, _MANAGED_HANDLER_ATTRIBUTE, True)
        root_logger.addHandler(console_handler)

    logging.captureWarnings(True)
    return log_path


def install_exception_hooks(
    *,
    logger_name: str = "depth3d.crash",
) -> tuple[Callable[..., object], Callable[..., object] | None]:
    """Install system and thread exception hooks that preserve normal stderr.

    Returns the installed hooks to make the behaviour directly testable.
    Repeated calls are safe because the newly installed hook delegates to the
    original interpreter hook, not to a previously installed Depth3D hook.
    """

    logger = logging.getLogger(logger_name)
    original_sys_hook = getattr(sys, "__excepthook__", sys.excepthook)

    def _sys_hook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: TracebackType | None,
    ) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            original_sys_hook(exc_type, exc_value, exc_traceback)
            return
        logger.critical(
            "Uncaught exception in the application process",
            exc_info=(exc_type, exc_value, exc_traceback),
        )
        original_sys_hook(exc_type, exc_value, exc_traceback)

    setattr(_sys_hook, "_depth3d_exception_hook", True)
    sys.excepthook = _sys_hook

    thread_hook: Callable[..., object] | None = None
    if hasattr(threading, "excepthook"):
        original_thread_hook = getattr(threading, "__excepthook__", threading.excepthook)

        def _thread_hook(args: threading.ExceptHookArgs) -> None:
            logger.critical(
                "Uncaught exception in thread %s",
                getattr(args.thread, "name", "unknown"),
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
            original_thread_hook(args)

        setattr(_thread_hook, "_depth3d_exception_hook", True)
        threading.excepthook = _thread_hook
        thread_hook = _thread_hook

    return _sys_hook, thread_hook



def _select_log_directory(requested: Path) -> Path:
    """Choose a writable log directory without making startup fragile."""

    candidates = [
        requested,
        Path.home() / ".local" / "state" / "depth3d-model-lab" / "logs",
        Path(tempfile.gettempdir()) / "depth3d-model-lab" / "logs",
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            resolved = candidate.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            resolved.mkdir(parents=True, exist_ok=True)
            probe = resolved / f".write-test-{os.getpid()}"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return resolved
        except OSError:
            continue
    raise OSError("No writable directory is available for application logs.")

def _remove_managed_handlers(logger: logging.Logger) -> None:
    for handler in tuple(logger.handlers):
        if getattr(handler, _MANAGED_HANDLER_ATTRIBUTE, False):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass


def _resolve_level(value: str | int) -> int:
    if isinstance(value, int):
        return value
    name = str(value).strip().upper()
    resolved = logging.getLevelName(name)
    return resolved if isinstance(resolved, int) else logging.INFO


def _safe_process_name(value: str) -> str:
    normalized = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(value).strip().lower()
    ).strip("_")
    return normalized or "app"
