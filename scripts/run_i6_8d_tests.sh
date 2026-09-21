#!/usr/bin/env bash
# I6.8D: passive sidecar transport, controller wiring and focused predecessor checks.
# Run from repository root; never starts SfM, PatchMatch or fusion.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2

python -m py_compile \
  core/reconstruction/reconstruction_job_telemetry_contract.py \
  core/reconstruction/reconstruction_job_progress_observer.py \
  core/reconstruction/reconstruction_job_resource_sampler.py \
  core/reconstruction/reconstruction_job_telemetry_transport.py \
  app/reconstruction_job_telemetry_relay.py \
  app/offline_reconstruction_controller.py \
  app/reconstruction_dense_job_controller.py \
  tests/unit/test_i6_8d_job_telemetry_transport.py \
  tests/unit/test_i6_8d_controller_wiring_static.py \
  tests/ui/test_i6_8d_job_telemetry_relay_ui.py
compile_rc=$?
echo "I6_8D_COMPILE_RC=$compile_rc"

python -m pytest -q \
  tests/unit/test_i6_8d_job_telemetry_transport.py \
  tests/unit/test_i6_8d_controller_wiring_static.py \
  tests/ui/test_i6_8d_job_telemetry_relay_ui.py \
  tests/unit/test_i6_8c_job_resource_sampler.py \
  tests/unit/test_i6_8b_job_progress_observer.py \
  tests/unit/test_i6_8a_job_telemetry_contract.py \
  tests/integration/test_i6_7e2b2a_isolated_dense_worker.py \
  tests/integration/test_i6_7e2b_explicit_final_dense_admission.py
pytest_rc=$?
echo "I6_8D_TARGETED_RC=$pytest_rc"

bash -n scripts/run_i6_8d_tests.sh
bash_rc=$?
echo "I6_8D_BASH_RC=$bash_rc"

if [[ "$compile_rc" -eq 0 && "$pytest_rc" -eq 0 && "$bash_rc" -eq 0 ]]; then
  echo "I6_8D_TEST_RC=0"
  exit 0
fi
echo "I6_8D_TEST_RC=1" >&2
exit 1
