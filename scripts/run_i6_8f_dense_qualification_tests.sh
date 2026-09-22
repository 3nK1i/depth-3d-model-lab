#!/usr/bin/env bash
# I6.8F: bounded non-MVS dense-worker wrapper -> resource sampler -> Qt page.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2
python -m py_compile \
  scripts/run_i6_8f_dense_telemetry_fixture.py \
  tests/ui/test_i6_8f_dense_telemetry_qualification.py \
  tests/unit/test_i6_8f_dense_telemetry_fixture_safety.py
compile_rc=$?
echo "I6_8F_DENSE_COMPILE_RC=$compile_rc"
python -c 'import PyQt6, pytestqt, psutil' >/dev/null 2>&1
qt_rc=$?
echo "I6_8F_DENSE_QT_DEPENDENCIES_RC=$qt_rc"
if [[ "$compile_rc" -ne 0 || "$qt_rc" -ne 0 ]]; then
  echo 'I6_8F_DENSE_QUALIFICATION_RC=1' >&2
  exit 1
fi
python -m pytest -q \
  tests/unit/test_i6_8f_dense_telemetry_fixture_safety.py \
  tests/ui/test_i6_8f_dense_telemetry_qualification.py \
  tests/ui/test_i6_8f_main_page_telemetry_integration.py \
  tests/unit/test_i6_8b_job_progress_observer.py
pytest_rc=$?
echo "I6_8F_DENSE_QT_TEST_RC=$pytest_rc"
bash -n scripts/run_i6_8f_dense_qualification_tests.sh
bash_rc=$?
echo "I6_8F_DENSE_BASH_RC=$bash_rc"
if [[ "$pytest_rc" -eq 0 && "$bash_rc" -eq 0 ]]; then
  echo 'I6_8F_DENSE_QUALIFICATION_RC=0'
  exit 0
fi
echo 'I6_8F_DENSE_QUALIFICATION_RC=1' >&2
exit 1
