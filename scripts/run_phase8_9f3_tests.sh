#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mode="${1:-quick}"
case "$mode" in
  quick)
    pytest -q \
      tests/unit/test_phase8_9f3_real_gaussian_campaign.py \
      tests/integration/test_phase8_9f3_real_gaussian_contract.py
    ;;
  affected)
    pytest -q \
      tests/unit/test_phase8_9f3_real_gaussian_campaign.py \
      tests/integration/test_phase8_9f3_real_gaussian_contract.py \
      tests/unit/test_reconstruction_gaussian_splatting.py \
      tests/unit/test_reconstruction_gaussian_backends.py \
      tests/unit/test_reconstruction_gaussian_benchmarking.py \
      tests/unit/test_reconstruction_gaussian_benchmark_quality.py \
      tests/unit/test_reconstruction_gaussian_benchmark_performance.py \
      tests/unit/test_reconstruction_pipeline_appearance_f4.py \
      tests/unit/test_phase8_7f4_backward_compatibility.py \
      tests/integration/test_reconstruction_pipeline_appearance_f4.py
    ;;
  *)
    echo "usage: $0 {quick|affected}" >&2
    exit 2
    ;;
esac
