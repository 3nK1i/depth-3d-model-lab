#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.." || exit 2
python -m py_compile core/reconstruction/reconstruction_method_selection_contract.py \
  tests/unit/test_i6_8_5b_reconstruction_method_contract.py
COMPILE_RC=$?
echo "I6_8_5B_COMPILE_RC=$COMPILE_RC"
python -m pytest -q tests/unit/test_i6_8_5b_reconstruction_method_contract.py \
  tests/unit/test_i6_4a_provider_availability_selection_contract.py
TEST_RC=$?
echo "I6_8_5B_TEST_RC=$TEST_RC"
if [ "$COMPILE_RC" -ne 0 ] || [ "$TEST_RC" -ne 0 ]; then exit 1; fi
echo 'I6_8_5B_VALIDATION=PASS'
