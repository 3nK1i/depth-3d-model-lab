#!/usr/bin/env bash
# I6.8F main-application Qt integration: diagnostic sidecars only; no worker launch.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2
python -m py_compile tests/ui/test_i6_8f_main_page_telemetry_integration.py
compile_rc=$?
echo "I6_8F_MAIN_PAGE_COMPILE_RC=$compile_rc"
python -m pytest -q tests/ui/test_i6_8f_main_page_telemetry_integration.py tests/ui/test_i6_8e_job_telemetry_ui.py tests/ui/test_i6_8d_job_telemetry_relay_ui.py
pytest_rc=$?
echo "I6_8F_MAIN_PAGE_QT_RC=$pytest_rc"
bash -n scripts/run_i6_8f_main_page_ui_tests.sh
bash_rc=$?
echo "I6_8F_MAIN_PAGE_BASH_RC=$bash_rc"
if [[ "$compile_rc" -eq 0 && "$pytest_rc" -eq 0 && "$bash_rc" -eq 0 ]]; then
  echo 'I6_8F_MAIN_PAGE_UI_TEST_RC=0'
  exit 0
fi
echo 'I6_8F_MAIN_PAGE_UI_TEST_RC=1' >&2
exit 1
