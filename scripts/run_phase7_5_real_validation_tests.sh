#!/usr/bin/env bash
set -euo pipefail
python -m pytest -q tests/unit/test_real_measurement_validation_script.py
