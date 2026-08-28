#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

metric3d_backward_focused=(
  tests/unit/test_metric3d_v2_adapter.py
  tests/unit/test_metric3d_v2_scripts.py
  tests/unit/test_model_output_contract.py
  tests/unit/test_model_registry.py
)
metric3d_backward_stress=(tests/stress/test_metric3d_v2_adapter_stress.py)
i43e1_focused=(
  tests/unit/test_i4_3e1_metric3d_v2_dense_provider.py
  tests/integration/test_i4_3e1_metric3d_v2_provider_contract.py
)
i43e1_stress=(tests/stress/test_i4_3e1_metric3d_v2_provider_stress.py)

case "${1:-all}" in
  focused)
    python -m pytest -q "${metric3d_backward_focused[@]}" "${i43e1_focused[@]}"
    ;;
  stress)
    python -m pytest -q "${metric3d_backward_stress[@]}" "${i43e1_stress[@]}"
    ;;
  all)
    bash scripts/run_i4_3d3_tests.sh all
    python -m pytest -q \
      "${metric3d_backward_focused[@]}" "${i43e1_focused[@]}" \
      "${metric3d_backward_stress[@]}" "${i43e1_stress[@]}"
    ;;
  *)
    echo "usage: $0 [focused|stress|all]" >&2
    exit 2
    ;;
esac

python - <<'PYPROBE'
import importlib.util
import sys
from pathlib import Path
from core.models.adapters.metric3d_v2_adapter import Metric3DV2GeometryModel
from core.models.catalog import METRIC3D_V2_MODEL_ID

root = Path.cwd()
source = root / "third_party/Metric3D/.eia_metric3d_v2_ready.json"
worker = root / "scripts/reconstruction_single_image_depth_provider_worker.py"
assert Metric3DV2GeometryModel is not None
assert METRIC3D_V2_MODEL_ID == "metric3d-v2"
assert source.is_file()

before = set(sys.modules)
spec = importlib.util.spec_from_file_location("i43e1_worker_script_probe", worker)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
new_modules = set(sys.modules) - before
for forbidden in ("torch", "cv2", "core.models.registry"):
    assert forbidden not in new_modules, f"worker imported {forbidden} at module load"
assert callable(module.main) and callable(module.run)
print("I4.3E1 Metric3D adapter/source + lazy worker import probe: PASS")
PYPROBE
