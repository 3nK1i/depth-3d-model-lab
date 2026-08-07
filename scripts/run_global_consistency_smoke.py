#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reconstruction import (
    DeterministicMockPoseGraphOptimizer,
    RelocalizationAcceptancePolicy,
    RelocalizationStatus,
)
from tests.reconstruction_global_consistency_helpers import make_drifted_graph


def main() -> int:
    graph = make_drifted_graph()
    result = DeterministicMockPoseGraphOptimizer().optimize(graph)
    final_pose = result.optimized_poses[-1]
    relocalization = RelocalizationAcceptancePolicy().evaluate(
        query_frame_id="smoke-query",
        map_revision=result.output_revision,
        matched_keyframe_id=final_pose.keyframe_id,
        confidence=0.95,
        inlier_count=96,
        T_reconstruction_from_camera=final_pose.T_reconstruction_from_camera,
    )

    positions = [float(p.T_reconstruction_from_camera[0, 3]) for p in result.optimized_poses]
    passed = (
        result.converged
        and result.final_error < result.initial_error
        and np.allclose(positions, [0.0, 1.0, 2.0, 3.0])
        and result.gauge.unit_to_metre is None
        and relocalization.status is RelocalizationStatus.SUCCESS
        and relocalization.gauge.unit_to_metre is None
    )
    print(f"Phase 8.5A global consistency foundation smoke: {'PASS' if passed else 'FAIL'}")
    print(f"input_revision={result.input_revision} output_revision={result.output_revision}")
    print(f"loop_closures={result.loop_closure_count} initial_error={result.initial_error:.6f} final_error={result.final_error:.6f}")
    print(f"optimized_x={positions}")
    print("raw_pose_layer=tracking optimized_pose_layer=optimized")
    print("pose_convention=T_reconstruction_from_camera")
    print("gauge=arbitrary unit_to_metre=None")
    print(f"relocalization={relocalization.status.value} matched_keyframe={relocalization.matched_keyframe_id}")
    print("native_loop_closure_enabled=False native_global_optimization_enabled=False")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
