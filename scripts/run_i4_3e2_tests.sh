#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

unidepth_backward_focused=(
  tests/unit/test_unidepth_v2_adapter.py
  tests/unit/test_unidepth_v2_scripts.py
  tests/unit/test_model_output_contract.py
  tests/unit/test_model_registry.py
  tests/unit/test_gpu_profiles.py
)
unidepth_backward_stress=(tests/stress/test_unidepth_v2_adapter_stress.py)
i43e2_focused=(
  tests/unit/test_i4_3e2_unidepth_v2_dense_provider.py
  tests/integration/test_i4_3e2_unidepth_v2_provider_contract.py
)
i43e2_stress=(tests/stress/test_i4_3e2_unidepth_v2_provider_stress.py)

case "${1:-all}" in
  focused)
    python -m pytest -q "${unidepth_backward_focused[@]}" "${i43e2_focused[@]}"
    ;;
  stress)
    python -m pytest -q "${unidepth_backward_stress[@]}" "${i43e2_stress[@]}"
    ;;
  all)
    bash scripts/run_i4_3e1_tests.sh all
    python -m pytest -q \
      "${unidepth_backward_focused[@]}" "${i43e2_focused[@]}" \
      "${unidepth_backward_stress[@]}" "${i43e2_stress[@]}"
    ;;
  *)
    echo "usage: $0 [focused|stress|all]" >&2
    exit 2
    ;;
esac

python - <<'PYPROBE'
import importlib.util
import json
import sys
from pathlib import Path
from core.models.adapters.unidepth_v2_adapter import UniDepthV2GeometryModel
from core.models.catalog import UNIDEPTH_V2_MODEL_ID

root = Path.cwd()
marker = root / "third_party/UniDepth/.eia_unidepth_v2_compat.json"
worker = root / "scripts/reconstruction_unidepth_v2_provider_worker.py"
assert UniDepthV2GeometryModel is not None
assert UNIDEPTH_V2_MODEL_ID == "unidepth-v2"
assert marker.is_file()
payload = json.loads(marker.read_text(encoding="utf-8"))
assert payload.get("commit") == "8d8cfe4c7ee15297099983607febf0d4f32eb3d6"

before = set(sys.modules)
spec = importlib.util.spec_from_file_location("i43e2_worker_script_probe", worker)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
new_modules = set(sys.modules) - before
for forbidden in ("torch", "cv2", "unidepth", "core.models.registry"):
    assert forbidden not in new_modules, f"worker imported {forbidden} at module load"
assert callable(module.main) and callable(module.run)
print("I4.3E2 UniDepthV2 adapter/source pin + lazy worker import probe: PASS")
PYPROBE
