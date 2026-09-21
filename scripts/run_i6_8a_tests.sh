#!/usr/bin/env bash
# Narrow I6.8A contract and immediate predecessor validation. No reconstruction runs.
set -uo pipefail

python -m py_compile \
  core/reconstruction/reconstruction_job_telemetry_contract.py \
  tests/unit/test_i6_8a_job_telemetry_contract.py
compile_rc=$?
echo "I6_8A_COMPILE_RC=$compile_rc"

python -m pytest -q \
  tests/unit/test_i6_8a_job_telemetry_contract.py \
  tests/unit/test_i6_7a_progressive_dense_snapshot_contract.py \
  tests/unit/test_i6_6a_progressive_sfm_snapshot_contract.py
pytest_rc=$?
echo "I6_8A_CONTRACT_TEST_RC=$pytest_rc"

if [[ "$compile_rc" -eq 0 && "$pytest_rc" -eq 0 ]]; then
  echo 'I6_8A_TEST_RC=0'
  exit 0
fi
printf 'I6_8A_TEST_RC=1\n' >&2
exit 1
