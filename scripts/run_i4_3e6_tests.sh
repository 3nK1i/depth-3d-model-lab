#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

predecessor_sanity=(
  tests/unit/test_i4_3e5_depth_anything_v2_dense_provider.py
  tests/integration/test_i4_3e5_depth_anything_v2_provider_contract.py
)
marigold_backward=(
  tests/unit/test_marigold_adapter.py
  tests/unit/test_marigold_scripts.py
  tests/unit/test_model_output_contract.py
  tests/unit/test_model_registry.py
  tests/unit/test_gpu_profiles.py
  tests/unit/test_hardware_tuning_matrix.py
)
marigold_backward_stress=(tests/stress/test_marigold_adapter_stress.py)
i43e6_focused=(
  tests/unit/test_i4_3e6_marigold_dense_provider.py
  tests/integration/test_i4_3e6_marigold_provider_contract.py
)
i43e6_stress=(tests/stress/test_i4_3e6_marigold_provider_stress.py)

case "${1:-all}" in
  focused)
    python -m pytest -q "${marigold_backward[@]}" "${i43e6_focused[@]}"
    ;;
  stress)
    python -m pytest -q "${marigold_backward_stress[@]}" "${i43e6_stress[@]}"
    ;;
  all)
    python -m pytest -q \
      "${predecessor_sanity[@]}" \
      "${marigold_backward[@]}" \
      "${marigold_backward_stress[@]}" \
      "${i43e6_focused[@]}" \
      "${i43e6_stress[@]}"
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
from core.models.adapters.marigold_adapter import MarigoldGeometryModel
from core.models.catalog import MARIGOLD_MODEL_ID

root = Path.cwd()
worker = root / "scripts/reconstruction_marigold_provider_worker.py"
assert MarigoldGeometryModel is not None
assert MARIGOLD_MODEL_ID == "marigold"

before = set(sys.modules)
spec = importlib.util.spec_from_file_location("i43e6_marigold_worker_probe", worker)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
new_modules = set(sys.modules) - before
for forbidden in ("torch", "cv2", "diffusers", "core.models.registry"):
    assert forbidden not in new_modules, f"worker imported {forbidden} at module load"
assert callable(module.main) and callable(module.run)
print("I4.3E6 Marigold adapter + lazy worker import probe: PASS")
PYPROBE
