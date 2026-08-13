#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-quick}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXPECTED_BRANCH="phase8-reconstruction"
EXPECTED_HEAD="dff62357874e28de8d0ceadb9a222dd0cd17d999"
EXPECTED_SCOPE=$(cat <<'SCOPE'
README.md
RELEASE_NOTES_v0.8.0.md
docs/phase_history/PHASE8_9H1_RELEASE_INVENTORY_AUDIT.md
docs/phase_history/PHASE8_9H2_RELEASE_PREPARATION.md
scripts/run_phase8_9h2_release_prep_tests.sh
tests/unit/test_phase8_9h2_release_prep_contract.py
SCOPE
)

check_repo() {
    test "$(git branch --show-current)" = "$EXPECTED_BRANCH" || {
        echo "FAIL: wrong branch" >&2
        exit 1
    }
    test "$(git rev-parse HEAD)" = "$EXPECTED_HEAD" || {
        echo "FAIL: unexpected H2 baseline HEAD" >&2
        exit 1
    }
    remote_head=$(git ls-remote origin refs/heads/phase8-reconstruction | awk 'NR==1 {print $1}')
    test "$remote_head" = "$EXPECTED_HEAD" || {
        echo "FAIL: origin/phase8-reconstruction moved during H2" >&2
        exit 1
    }
    git diff --check

    if git rev-parse -q --verify refs/tags/v0.8.0 >/dev/null; then
        echo "FAIL: v0.8.0 tag must remain absent during H2" >&2
        exit 1
    fi
    if [ -n "$(git ls-remote --tags origin 'refs/tags/v0.8.0' 'refs/tags/v0.8.0^{}')" ]; then
        echo "FAIL: remote v0.8.0 tag must remain absent during H2" >&2
        exit 1
    fi

    actual=$(git status --porcelain=v1 | sed -E 's/^.. //' | sort)
    expected=$(printf '%s\n' "$EXPECTED_SCOPE" | sort)
    if [ "$actual" != "$expected" ]; then
        echo "FAIL: H2 working-tree scope mismatch" >&2
        echo "--- expected ---" >&2
        printf '%s\n' "$expected" >&2
        echo "--- actual ---" >&2
        printf '%s\n' "$actual" >&2
        exit 1
    fi
}

check_metadata() {
    grep -Fq '__version__ = "0.7.5"' core/version.py
    grep -Fq 'RELEASE_NAME = "Phase 7.5 Stable"' core/version.py
    grep -Fq 'RELEASE_PHASE = 7.5' core/version.py
    grep -Fq 'version = "0.7.5"' pyproject.toml
    grep -Fq 'version: "0.7.5"' config/app.yaml
    grep -Fq 'release: "Phase 7.5 Stable"' config/app.yaml
}

case "$MODE" in
    quick)
        check_repo
        check_metadata
        python -m pytest -q tests/unit/test_phase8_9h2_release_prep_contract.py
        ;;
    affected)
        check_repo
        check_metadata
        python -m pytest -q tests/unit/test_phase8_9h2_release_prep_contract.py
        python -m pytest -q \
            tests/unit/test_release_metadata.py \
            tests/unit/test_phase6_release_metadata.py \
            tests/unit/test_phase7_release_metadata.py \
            tests/unit/test_phase75_release_gate.py
        check_repo
        ;;
    *)
        echo "usage: $0 {quick|affected}" >&2
        exit 2
        ;;
esac

echo "Phase 8.9H2 release preparation: PASS ($MODE)"
