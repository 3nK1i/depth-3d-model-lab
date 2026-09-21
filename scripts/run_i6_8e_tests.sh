#!/usr/bin/env bash
# I6.8E passive UI integration; no SfM/MVS starts.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2
python -m py_compile \
  app/reconstruction_job_telemetry_presentation.py \
  app/widgets/reconstruction_page.py \
  tests/unit/test_i6_8e_job_telemetry_presentation.py \
  tests/unit/test_i6_8e_page_binding_static.py \
  tests/ui/test_i6_8e_job_telemetry_ui.py
compile_rc=$?
echo "I6_8E_COMPILE_RC=$compile_rc"
python -m pytest -q \
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
echo "I6_8E_TARGETED_RC=$pytest_rc"
bash -n scripts/run_i6_8e_tests.sh
bash_rc=$?
echo "I6_8E_BASH_RC=$bash_rc"
if [[ "$compile_rc" -eq 0 && "$pytest_rc" -eq 0 && "$bash_rc" -eq 0 ]]; then
  echo "I6_8E_TEST_RC=0"
  exit 0
fi
echo "I6_8E_TEST_RC=1" >&2
exit 1
