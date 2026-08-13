#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-quick}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

run_strict_repo_gate() {
    python scripts/run_phase8_9h3_release_candidate_qualification.py
}

case "$MODE" in
    quick)
        run_strict_repo_gate
        python -m pytest -q \
            tests/unit/test_phase8_9h2_release_prep_contract.py \
            tests/unit/test_phase8_9h3_release_candidate_contract.py
        run_strict_repo_gate
        ;;
    affected)
        run_strict_repo_gate
        python -m compileall -q \
            scripts/run_phase8_9h3_release_candidate_qualification.py \
            tests/unit/test_phase8_9h2_release_prep_contract.py \
            tests/unit/test_phase8_9h3_release_candidate_contract.py
        bash -n scripts/run_phase8_9h2_release_prep_tests.sh
        bash -n scripts/run_phase8_9h3_release_candidate_tests.sh
        python -m pytest -q \
            tests/unit/test_phase8_9h2_release_prep_contract.py \
            tests/unit/test_phase8_9h3_release_candidate_contract.py
        python -m pytest -q \
            tests/unit/test_release_metadata.py \
            tests/unit/test_phase6_release_metadata.py \
            tests/unit/test_phase7_release_metadata.py \
            tests/unit/test_phase75_release_gate.py \
            tests/ui/test_release_header.py
        run_strict_repo_gate
        ;;
    *)
        echo "usage: $0 {quick|affected}" >&2
        exit 2
        ;;
esac

echo "Phase 8.9H3 release-candidate qualification: PASS ($MODE)"
