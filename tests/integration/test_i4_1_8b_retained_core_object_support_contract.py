from __future__ import annotations

from pathlib import Path


def test_i4_1_8b_runner_reconstructs_attribution_without_heavy_reconstruction_reruns() -> None:
    root = Path(__file__).resolve().parents[2]
    runner = (root / 'scripts/run_i4_1_8b_retained_core_object_support_isolation.py').read_text(encoding='utf-8')
    assert '_aggregate_saved_e1_source_frame_voxels' in runner
    assert 'exact retained-Core view attribution: PASS' in runner
    assert 'run_f_confidence_artifact_filtering.py' not in runner
    assert 'run_g_persistent_dense_rgb_geometry.py' not in runner
    assert 'run_i2_full_video_quality_qualification.py' not in runner
    assert 'sfm_rerun=False da3_rerun=False dense_pair_rerun=False fusion_rerun=False' in runner
    assert 'production_target_changed=False orbit_thresholds_changed=False' in runner
    assert 'metric_scale_claimed=False measurement_authoritative=False surface_mesh_generated=False' in runner


def test_i4_1_8b_component_selection_explicitly_excludes_orbit_gate_feedback() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / 'core/reconstruction/offline_dense_i4_retained_core_object_support.py').read_text(encoding='utf-8')
    assert 'component_selection_uses_orbit_metrics": False' in text
    assert 'component_selection_uses_locked_i4_1_gates": False' in text
    assert 'posthoc_orbit_used_for_selection_or_readiness": False' in text
    assert 'authorizes_target_change": False' in text
    assert 'production_target_changed": False' in text


def test_i4_1_8b_preview_uses_reconstruction_canvas_and_isolated_support_overlay() -> None:
    root = Path(__file__).resolve().parents[2]
    text = (root / 'scripts/run_i4_1_8b_object_support_preview.py').read_text(encoding='utf-8')
    assert 'ReconstructionCanvas' in text
    assert 'set_dense_rgb_points' in text
    assert 'update_sparse_landmarks' in text
    assert 'isolated_core_points_reconstruction' in text
    assert 'green=isolated retained Core support' in text


def test_i4_1_8b_exact_core_attribution_replays_saved_source_voxels(tmp_path) -> None:
    import numpy as np
    from scripts.run_i4_1_8b_retained_core_object_support_isolation import _exact_core_view_attribution

    voxel = 0.1
    core = np.asarray([[0.05,0.05,1.05],[0.15,0.05,1.05],[0.25,0.05,1.05]], dtype=np.float64)
    pose = np.eye(4, dtype=np.float64)
    sources = np.asarray([10,20], dtype=np.int64)
    frames = []
    source_sets = ([0,1],[1,2])
    for rank, members in enumerate(source_sets):
        pts = core[np.asarray(members, dtype=np.int64)]
        # Identity pose and z-depth reproduce canonical points exactly.
        depth = np.full((1, len(pts)), np.nan, dtype=np.float64)
        depth[0,:] = pts[:,2]
        cloud = tmp_path / f'cloud_{rank}.npz'
        np.savez_compressed(
            cloud,
            points_reconstruction=pts,
            colors_rgb=np.full((len(pts),3), 100, dtype=np.uint8),
            aligned_depth_reconstruction_units=depth,
            T_reconstruction_from_camera=pose,
        )
        frames.append({'workspace_order':rank,'source_sequence_index':int(sources[rank]),'cloud_npz':str(cloud)})
    matrix, records = _exact_core_view_attribution(
        core_points=core, voxel_size=voxel, frames=frames, expected_sources=sources,
    )
    assert matrix.tolist() == [[True,True,False],[False,True,True]]
    assert [r['retained_core_observations'] for r in records] == [2,2]
