#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

predecessor_sanity=(
  tests/unit/test_i4_3e2_unidepth_v2_dense_provider.py
  tests/integration/test_i4_3e2_unidepth_v2_provider_contract.py
)
moge_backward=(
  tests/unit/test_moge2_adapter.py
  tests/unit/test_model_output_contract.py
  tests/unit/test_model_registry.py
  tests/unit/test_gpu_profiles.py
  tests/unit/test_hardware_tuning_matrix.py
)
i43e3_focused=(
  tests/unit/test_i4_3e3_moge2_dense_provider.py
  tests/integration/test_i4_3e3_moge2_provider_contract.py
)
i43e3_stress=(tests/stress/test_i4_3e3_moge2_provider_stress.py)

case "${1:-all}" in
  focused)
    python -m pytest -q "${moge_backward[@]}" "${i43e3_focused[@]}"
    ;;
  stress)
    python -m pytest -q "${i43e3_stress[@]}"
    ;;
  all)
    python -m pytest -q \
      "${predecessor_sanity[@]}" \
      "${moge_backward[@]}" \
      "${i43e3_focused[@]}" \
      "${i43e3_stress[@]}"
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
from core.models.adapters.moge2_adapter import Moge2GeometryModel
from core.models.catalog import MOGE2_MODEL_ID

root = Path.cwd()
worker = root / "scripts/reconstruction_moge2_provider_worker.py"
assert Moge2GeometryModel is not None
assert MOGE2_MODEL_ID == "moge-2"

before = set(sys.modules)
spec = importlib.util.spec_from_file_location("i43e3_worker_script_probe", worker)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
new_modules = set(sys.modules) - before
for forbidden in ("torch", "cv2", "moge", "core.models.registry"):
    assert forbidden not in new_modules, f"worker imported {forbidden} at module load"
assert callable(module.main) and callable(module.run)
print("I4.3E3 MoGe-2 adapter + lazy worker import probe: PASS")
PYPROBE
