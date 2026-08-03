"""Rotating-log and exception-hook reliability tests."""

from __future__ import annotations

import logging
import sys
import threading

from core.runtime.logging_setup import configure_logging, install_exception_hooks


def _managed_handlers() -> list[logging.Handler]:
    return [
        handler
        for handler in logging.getLogger().handlers
        if getattr(handler, "_depth3d_managed_handler", False)
    ]


def test_logging_configuration_is_idempotent_and_writes_file(tmp_path) -> None:
    root = logging.getLogger()
    original_level = root.level
    try:
        first = configure_logging(process_name="unit", log_dir=tmp_path, console=False)
        logging.getLogger("depth3d.test").warning("first diagnostic")
        for handler in _managed_handlers():
            handler.flush()

        second = configure_logging(process_name="unit", log_dir=tmp_path, console=False)
        logging.getLogger("depth3d.test").error("second diagnostic")
        for handler in _managed_handlers():
            handler.flush()

        assert first == second == tmp_path / "unit.log"
        assert len(_managed_handlers()) == 1
        text = second.read_text(encoding="utf-8")
        assert "second diagnostic" in text
    finally:
        for handler in _managed_handlers():
            root.removeHandler(handler)
            handler.close()
        root.setLevel(original_level)


def test_exception_hooks_are_installed_and_marked(monkeypatch) -> None:
    original_sys = sys.excepthook
    original_thread = threading.excepthook
    try:
        sys_hook, thread_hook = install_exception_hooks(logger_name="depth3d.test.crash")
        assert sys.excepthook is sys_hook
        assert getattr(sys_hook, "_depth3d_exception_hook", False)
        assert thread_hook is None or getattr(thread_hook, "_depth3d_exception_hook", False)
    finally:
        sys.excepthook = original_sys
        threading.excepthook = original_thread
