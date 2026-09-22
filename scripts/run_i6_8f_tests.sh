#!/usr/bin/env bash
# I6.8F qualification checks: no reconstruction jobs are started.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2
python -m py_compile \
  scripts/run_i6_8f_telemetry_qualification.py \
  scripts/run_i6_8f_real_worker_qualification.py \
  tests/unit/test_i6_8f_telemetry_qualification.py \
  tests/unit/test_i6_8f_real_worker_qualification.py
compile_rc=$?
echo "I6_8F_COMPILE_RC=$compile_rc"
python -m pytest -q \
  tests/unit/test_i6_8f_telemetry_qualification.py \
  tests/unit/test_i6_8f_real_worker_qualification.py \
  tests/unit/test_i6_8e_job_telemetry_presentation.py \
  tests/unit/test_i6_8e_page_binding_static.py \
  tests/ui/test_i6_8e_job_telemetry_ui.py \
  tests/unit/test_i6_8d_job_telemetry_transport.py \
  tests/unit/test_i6_8d_controller_wiring_static.py \
  tests/ui/test_i6_8d_job_telemetry_relay_ui.py \
  tests/unit/test_i6_8c_job_resource_sampler.py \
  tests/unit/test_i6_8b_job_progress_observer.py \
  tests/unit/test_i6_8a_job_telemetry_contract.py
pytest_rc=$?
echo "I6_8F_TARGETED_RC=$pytest_rc"
bash -n scripts/run_i6_8f_tests.sh
bash_rc=$?
echo "I6_8F_BASH_RC=$bash_rc"
if [[ "$compile_rc" -eq 0 && "$pytest_rc" -eq 0 && "$bash_rc" -eq 0 ]]; then
  echo 'I6_8F_TEST_RC=0'
  exit 0
fi
echo 'I6_8F_TEST_RC=1' >&2
exit 1
