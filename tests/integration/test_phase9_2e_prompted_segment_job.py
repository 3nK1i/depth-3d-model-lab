from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _run(tmp_path: Path, spec: dict[str, object], name: str) -> tuple[dict[str, object], np.ndarray]:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[25:95, 35:125] = 180
    input_path = tmp_path / f"{name}-frame.npy"
    prompt_path = tmp_path / f"{name}-prompt.json"
    output = tmp_path / f"{name}-out"
    np.save(input_path, image)
    prompt_path.write_text(json.dumps(spec), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_prompted_segment_job.py"),
            "--input", str(input_path),
            "--output", str(output),
            "--prompt", str(prompt_path),
            "--model", "development-perception-mock",
            "--profile", "efficient",
            "--minimum-mask-pixels", "16",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "RESULT " in completed.stdout
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    mask = np.asarray(np.load(output / "mask.npy", allow_pickle=False), dtype=bool)
    return manifest, mask


def test_9_2e_isolated_worker_can_create_object_from_one_click(tmp_path: Path) -> None:
    manifest, mask = _run(
        tmp_path,
        {
            "operation": "create",
            "object_id": "interactive:abc",
            "parent_mask_id": "",
            "revision": 0,
            "positive_points": [[80, 60]],
            "negative_points": [],
        },
        "create",
    )
    assert manifest["operation"] == "create"
    assert manifest["object_id"] == "interactive:abc"
    assert manifest["parent_mask_id"] == ""
    assert manifest["revision"] == 0
    assert manifest["model_id"] == "development-perception-mock"
    assert mask.shape == (120, 160)
    assert np.any(mask)


def test_9_2e_isolated_worker_preserves_refinement_parent_and_revision(tmp_path: Path) -> None:
    manifest, mask = _run(
        tmp_path,
        {
            "operation": "refine",
            "object_id": "proposal:0",
            "parent_mask_id": "mask-r0",
            "revision": 1,
            "positive_points": [[80, 60], [86, 60]],
            "negative_points": [[80, 66]],
        },
        "refine",
    )
    assert manifest["operation"] == "refine"
    assert manifest["object_id"] == "proposal:0"
    assert manifest["parent_mask_id"] == "mask-r0"
    assert manifest["revision"] == 1
    assert len(manifest["positive_points"]) == 2
    assert len(manifest["negative_points"]) == 1
    assert np.any(mask)
