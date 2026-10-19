#!/usr/bin/env bash
# I6.7E1 isolated full-artifact read-only preflight. No provider execution/UI/Auto.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2
MODE="${1:-focused}"
if [[ "$MODE" != focused && "$MODE" != closure ]]; then
  echo 'Usage: bash scripts/run_i6_7e1_tests.sh focused|closure' >&2; exit 2
fi
python - <<'PY'
import hashlib
from pathlib import Path
checks = {
 'core/reconstruction/reconstruction_progressive_dense_contract.py': '918c67574ef342909fffdfd495d2503c5909bafd0624fb3856b34c488f2257ee',
 'core/reconstruction/reconstruction_progressive_dense_emission.py': '5c68ec59d1ec0238b15853e95135a56df7e2043824e9c835c66fadaa4f594ff7',
 'core/reconstruction/reconstruction_progressive_dense_transport.py': '5be5d9bca0b564c2454deccff5530b3f483ca7ae45425a4394c05a0efa968074',
 'app/reconstruction_progressive_dense_viewer.py': '453dbbdd8d90852ae0062e01c1370d14d55c849331cc157c551bf9490ab09dd8',
 'app/reconstruction_progressive_sfm_viewer.py': 'cda2e1da490a83a9af0875a95d4ac3672f9637b3dd616a7e6738911702864d76',
 'app/offline_reconstruction_controller.py': '467b551dc2a9f183d77f49f9768363b5057ca9af4adf2f7a26b5d35c54b5b70c',
 'app/reconstruction_controller.py': '57ef77cc5ebad69006b11ad25225e289ef540539e09e1636129b9d9bd4211186',
 'app/widgets/reconstruction_page.py': '2e5b406063f29cd02681fd299136c5836837dfbb60317536161748b62b09a590',
 'app/widgets/viewport/vispy_canvas.py': '2f960d58c2259884408e0a5d1813529eec47d84ec5fc0976282d40e5784d4a25',
 'tests/conftest.py': '923e20444d9e8c942a11f4503401f1e5cd845abae1e61f8b4e32f121be3df810',
}
for name, expected in checks.items():
 actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
 if actual != expected: raise SystemExit('I6_7E1_FROZEN_SOURCE_DRIFT=' + name)
print('I6_7E1_FROZEN_SOURCE=PASS')
PY
ANCHOR_RC=$?; echo "I6_7E1_ANCHOR_RC=$ANCHOR_RC"; if (( ANCHOR_RC != 0 )); then exit "$ANCHOR_RC"; fi
python -m py_compile core/reconstruction/reconstruction_dense_artifact_preflight.py scripts/run_i6_7e1_full_dense_artifact_probe.py tests/unit/test_i6_7e1_dense_artifact_preflight.py
COMPILE_RC=$?; echo "COMPILE_RC=$COMPILE_RC"; if (( COMPILE_RC != 0 )); then exit "$COMPILE_RC"; fi
python -m pytest -q tests/unit/test_i6_7e1_dense_artifact_preflight.py
FOCUSED_RC=$?; echo "I6_7E1_FOCUSED_RC=$FOCUSED_RC"; if (( FOCUSED_RC != 0 )); then exit "$FOCUSED_RC"; fi
if [[ "$MODE" == closure ]]; then
  python -m pytest -q tests/unit/test_i6_7d_progressive_dense_viewer_projection.py tests/stress/test_i6_7d_progressive_dense_viewer_projection_stress.py tests/unit/test_i6_7c_progressive_dense_transport.py tests/unit/test_i6_7c_progressive_dense_buffer.py tests/unit/test_i6_7b_progressive_dense_emission.py tests/unit/test_i6_7a_progressive_dense_snapshot_contract.py
  PREDECESSOR_RC=$?; echo "I6_7E1_PREDECESSOR_RC=$PREDECESSOR_RC"; if (( PREDECESSOR_RC != 0 )); then exit "$PREDECESSOR_RC"; fi
fi
echo 'I6_7E1_TEST_RC=0'
