from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_c4_native_worker_publishes_bounded_optimized_keyframe_poses() -> None:
    text = (ROOT / "native/live_slam_worker/main.cc").read_text(encoding="utf-8")
    assert 'phase8.8c4-worker-v1' in text
    assert 'kMaxLiveKeyframePoseSamples = 256' in text
    assert 'keyframes_reconstruction' in text
    assert 'keyframe->get_pose_wc()' in text
    assert 'T_reconstruction_from_camera' in text


def test_c4_worker_build_metadata_keeps_protocol_v1_and_records_keyframe_support() -> None:
    text = (ROOT / "scripts/build_stella_live_worker.sh").read_text(encoding="utf-8")
    assert 'worker_protocol_version=1' in text
    assert 'keyframe_pose_samples_supported=true' in text
    assert 'keyframe_pose_sample_cap=256' in text


def test_c4_does_not_pull_dense_or_neural_backends_into_live_viewer() -> None:
    files = [
        ROOT / "app/widgets/reconstruction_page.py",
        ROOT / "app/widgets/viewport/vispy_canvas.py",
        ROOT / "core/reconstruction/live_protocol.py",
    ]
    imported_modules: list[str] = []
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name.lower() for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_modules.append((node.module or "").lower())

    forbidden = ("depth_anything_3", "gsplat", "nerfstudio", "tinycudann")
    assert not any(
        module == prefix or module.startswith(prefix + ".")
        for module in imported_modules
        for prefix in forbidden
    )
