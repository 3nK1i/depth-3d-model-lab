#!/usr/bin/env bash
# I6.8C targeted Qt-free tests; does not execute any reconstruction or GPU workload.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2

python -m py_compile \
  core/reconstruction/reconstruction_job_telemetry_contract.py \
  core/reconstruction/reconstruction_job_progress_observer.py \
  core/reconstruction/reconstruction_job_resource_sampler.py \
  core/reconstruction/sfm_pycolmap.py \
  scripts/run_i6_7e2b2a_post_sparse_dense_worker.py \
  tests/unit/test_i6_8c_job_resource_sampler.py
compile_rc=$?
echo "I6_8C_COMPILE_RC=$compile_rc"

python -m pytest -q \
  tests/unit/test_i6_8c_job_resource_sampler.py \
  tests/unit/test_i6_8b_job_progress_observer.py \
  tests/unit/test_i6_8a_job_telemetry_contract.py \
  tests/unit/test_reconstruction_sfm_pycolmap.py \
  tests/unit/test_i4_3b_mvs_provider.py \
  tests/integration/test_i4_3b_mvs_provider_contract.py \
  tests/integration/test_i6_7e2b2a_isolated_dense_worker.py \
  tests/unit/test_i6_7e_production_dense_completion_verifier.py \
  tests/integration/test_i6_7e2b_explicit_final_dense_admission.py
pytest_rc=$?
echo "I6_8C_TARGETED_RC=$pytest_rc"

bash -n scripts/run_i6_8c_tests.sh
bash_rc=$?
echo "I6_8C_BASH_RC=$bash_rc"

if [[ "$compile_rc" -eq 0 && "$pytest_rc" -eq 0 && "$bash_rc" -eq 0 ]]; then
  echo "I6_8C_TEST_RC=0"
  exit 0
fi
printf 'I6_8C_TEST_RC=1\n' >&2
exit 1
