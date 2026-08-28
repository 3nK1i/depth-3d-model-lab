#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

predecessor_sanity=(
  tests/unit/test_i4_3e4_depth_pro_dense_provider.py
  tests/integration/test_i4_3e4_depth_pro_provider_contract.py
)
depth_anything_backward=(
  tests/unit/test_depth_anything_v2_adapter.py
  tests/unit/test_depth_anything_scripts.py
  tests/unit/test_model_output_contract.py
  tests/unit/test_model_registry.py
  tests/unit/test_gpu_profiles.py
  tests/unit/test_hardware_tuning_matrix.py
)
depth_anything_backward_stress=(tests/stress/test_depth_anything_adapters_stress.py)
i43e5_focused=(
  tests/unit/test_i4_3e5_depth_anything_v2_dense_provider.py
  tests/integration/test_i4_3e5_depth_anything_v2_provider_contract.py
)
i43e5_stress=(tests/stress/test_i4_3e5_depth_anything_v2_provider_stress.py)

case "${1:-all}" in
  focused)
    python -m pytest -q "${depth_anything_backward[@]}" "${i43e5_focused[@]}"
    ;;
  stress)
    python -m pytest -q "${depth_anything_backward_stress[@]}" "${i43e5_stress[@]}"
    ;;
  all)
    python -m pytest -q \
      "${predecessor_sanity[@]}" \
      "${depth_anything_backward[@]}" \
      "${depth_anything_backward_stress[@]}" \
      "${i43e5_focused[@]}" \
      "${i43e5_stress[@]}"
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
from core.models.adapters.depth_anything_v2_adapter import DepthAnythingV2GeometryModel
from core.models.catalog import DEPTH_ANYTHING_V2_MODEL_ID

root = Path.cwd()
worker = root / "scripts/reconstruction_depth_anything_v2_provider_worker.py"
assert DepthAnythingV2GeometryModel is not None
assert DEPTH_ANYTHING_V2_MODEL_ID == "depth-anything-v2"

before = set(sys.modules)
spec = importlib.util.spec_from_file_location("i43e5_depth_anything_v2_worker_probe", worker)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
new_modules = set(sys.modules) - before
for forbidden in ("torch", "cv2", "depth_anything_v2", "core.models.registry"):
    assert forbidden not in new_modules, f"worker imported {forbidden} at module load"
assert callable(module.main) and callable(module.run)
print("I4.3E5 Depth Anything V2 adapter + lazy worker import probe: PASS")
PYPROBE
