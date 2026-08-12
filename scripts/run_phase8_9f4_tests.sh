#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mode="${1:-quick}"
case "$mode" in
  quick)
    pytest -q \
      tests/unit/test_phase8_9f4_real_nerf_campaign.py \
      tests/integration/test_phase8_9f4_real_nerf_contract.py
    ;;
  affected)
    pytest -q \
      tests/unit/test_phase8_9f4_real_nerf_campaign.py \
      tests/integration/test_phase8_9f4_real_nerf_contract.py \
      tests/unit/test_reconstruction_nerf.py \
      tests/unit/test_reconstruction_nerf_e2.py \
      tests/unit/test_reconstruction_nerf_e3.py \
      tests/unit/test_reconstruction_nerf_e4.py \
      tests/unit/test_reconstruction_pipeline_appearance_f4.py \
      tests/unit/test_phase8_7f4_backward_compatibility.py \
      tests/integration/test_reconstruction_nerf_e2_qualification.py \
      tests/integration/test_reconstruction_nerf_e3_real_scene_validation.py \
      tests/integration/test_reconstruction_nerf_e4_performance_hardening.py \
      tests/integration/test_reconstruction_pipeline_appearance_f4.py
    ;;
  *)
    echo "usage: $0 {quick|affected}" >&2
    exit 2
    ;;
esac
