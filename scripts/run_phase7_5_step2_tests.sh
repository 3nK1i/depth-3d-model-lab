#!/usr/bin/env bash
set -euo pipefail
MODE="${1:-quick}"; PYTHON_BIN="${PYTHON:-python}"
STEP2_TESTS=(tests/unit/test_perception_raw_outputs.py tests/unit/test_perception_normalization.py tests/unit/test_perception_pipeline.py tests/unit/test_perception_catalog.py tests/unit/test_perception_availability.py tests/unit/test_perception_runtime_config.py tests/unit/test_perception_resource_planner.py tests/unit/test_perception_worker_messages.py tests/unit/test_ultralytics_perception_adapters.py tests/unit/test_phase75_step2_backward_compatibility.py)
compile(){ "$PYTHON_BIN" -m compileall -q app core scripts tests; }
quick(){ compile; "$PYTHON_BIN" -m pytest -q "${STEP2_TESTS[@]}"; }
integration(){ compile; "$PYTHON_BIN" -m pytest -q tests/integration/test_perception_worker_mock.py tests/integration/test_perception_process_lifecycle.py; }
full(){ local status=0; compile || status=1; for group in tests/unit tests/ui tests/integration tests/stress; do echo "==> pytest group: $group"; "$PYTHON_BIN" -m pytest -q "$group" || status=1; done; return "$status"; }
case "$MODE" in quick) quick;; integration) integration;; full) full;; *) echo "Usage: $0 {quick|integration|full}" >&2; exit 2;; esac
