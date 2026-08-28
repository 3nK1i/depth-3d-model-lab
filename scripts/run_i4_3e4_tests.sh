#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

predecessor_sanity=(
  tests/unit/test_i4_3e3_moge2_dense_provider.py
  tests/integration/test_i4_3e3_moge2_provider_contract.py
)
depth_pro_backward=(
  tests/unit/test_depth_pro_adapter.py
  tests/unit/test_depth_pro_checkpoint_script.py
  tests/unit/test_model_output_contract.py
  tests/unit/test_model_registry.py
  tests/unit/test_gpu_profiles.py
  tests/unit/test_hardware_tuning_matrix.py
)
depth_pro_backward_stress=(tests/stress/test_depth_pro_adapter_stress.py)
i43e4_focused=(
  tests/unit/test_i4_3e4_depth_pro_dense_provider.py
  tests/integration/test_i4_3e4_depth_pro_provider_contract.py
)
i43e4_stress=(tests/stress/test_i4_3e4_depth_pro_provider_stress.py)

case "${1:-all}" in
  focused)
    python -m pytest -q "${depth_pro_backward[@]}" "${i43e4_focused[@]}"
    ;;
  stress)
    python -m pytest -q "${depth_pro_backward_stress[@]}" "${i43e4_stress[@]}"
    ;;
  all)
    python -m pytest -q \
      "${predecessor_sanity[@]}" \
      "${depth_pro_backward[@]}" \
      "${depth_pro_backward_stress[@]}" \
      "${i43e4_focused[@]}" \
      "${i43e4_stress[@]}"
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
from core.models.adapters.depth_pro_adapter import DepthProGeometryModel
from core.models.catalog import DEPTH_PRO_MODEL_ID

root = Path.cwd()
worker = root / "scripts/reconstruction_depth_pro_provider_worker.py"
assert DepthProGeometryModel is not None
assert DEPTH_PRO_MODEL_ID == "depth-pro"

before = set(sys.modules)
spec = importlib.util.spec_from_file_location("i43e4_depth_pro_worker_script_probe", worker)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
new_modules = set(sys.modules) - before
for forbidden in ("torch", "cv2", "depth_pro", "core.models.registry"):
    assert forbidden not in new_modules, f"worker imported {forbidden} at module load"
assert callable(module.main) and callable(module.run)
print("I4.3E4 Depth Pro adapter + lazy worker import probe: PASS")
PYPROBE
