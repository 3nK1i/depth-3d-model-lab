#!/usr/bin/env bash
set +e
MODE="${1:-all}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2

run_pytest() {
  python -m pytest -q "$@"
}

RC=0
case "$MODE" in
  focused)
    run_pytest \
      tests/unit/test_i4_3g2b_matched_view.py \
      tests/integration/test_i4_3g2b_matched_view_contract.py
    RC=$?
    ;;
  stress)
    run_pytest tests/stress/test_i4_3g2b_matched_view_stress.py
    RC=$?
    ;;
  all)
    run_pytest \
      tests/unit/test_i4_3g2b_matched_view.py \
      tests/integration/test_i4_3g2b_matched_view_contract.py \
      tests/stress/test_i4_3g2b_matched_view_stress.py
    RC=$?
    ;;
  *)
    echo "Usage: $0 [focused|stress|all]" >&2
    exit 2
    ;;
esac

python - <<'PY'
from core.reconstruction.mvs_da3_matched_view import (
    MATCHED_VIEW_FRAMINGS,
    MATCHED_VIEW_REVIEW_DIMENSIONS,
    MATCHED_VIEW_VIEWS,
)
assert len(MATCHED_VIEW_VIEWS) == 4
assert tuple(MATCHED_VIEW_FRAMINGS) == ("detail", "context")
assert len(MATCHED_VIEW_REVIEW_DIMENSIONS) == 8
print("I4.3G2B matched-view contract/import probe: PASS")
PY
PROBE_RC=$?

if [ "$RC" -ne 0 ] || [ "$PROBE_RC" -ne 0 ]; then
  exit 1
fi
exit 0
