#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-quick}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export QT_OPENGL="${QT_OPENGL:-software}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"
export DEPTH3D_TEST_FAKE_VISPY="${DEPTH3D_TEST_FAKE_VISPY:-1}"

case "$MODE" in
  quick)
    python -m pytest -q tests/unit/test_phase9_6h_release_gate.py
    ;;
  integration)
    python -m pytest -q tests/integration/test_phase9_6h_release_gate_contract.py
    ;;
  backward)
    python -m pytest -q \
      tests/unit/test_phase9_6a_digital_twin_workspace.py \
      tests/unit/test_phase9_6b_digital_twin_scene_composition.py \
      tests/unit/test_phase9_6c_digital_twin_navigation.py \
      tests/unit/test_phase9_6d_digital_twin_interaction.py \
      tests/unit/test_phase9_6e0_estimated_metric_measurement.py \
      tests/unit/test_phase9_6e1_processing_jobs.py \
      tests/unit/test_phase9_6e2_da3_full_scene.py \
      tests/unit/test_phase9_6e3_nerf_build.py \
      tests/unit/test_phase9_6e4_workspace_ux_contract.py \
      tests/unit/test_phase9_6f_digital_twin_persistence.py \
      tests/unit/test_phase9_6g_digital_twin_resources.py \
      tests/integration/test_phase9_6f_digital_twin_persistence.py \
      tests/integration/test_phase9_6g_resource_coordination_contract.py
    ;;
  full)
    python -m compileall -q app core tests scripts main.py
    python -m pytest -q tests/unit
    python -m pytest -q tests/ui
    python -m pytest -q tests/integration
    python -m pytest -q tests/stress
    echo "Phase 9.6H current-tree full automated regression: PASS"
    echo "real_2_to_3_minute_scene_gate_must_be_run_separately=True"
    echo "photorealistic_quality_certified=False"
    echo "global_loop_closure_certified=False"
    echo "metric_scale_authoritative=False"
    echo "certified_metrology=False"
    echo "v0_9_0_tag_requires_explicit_post_H_approval=True"
    ;;
  *)
    echo "usage: $0 {quick|integration|backward|full}" >&2
    exit 2
    ;;
esac
