from pathlib import Path
import json
import subprocess
import sys


def test_f1_static_preflight_runs_without_external_backends(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    script = repo / "scripts" / "run_phase8_9f1_external_backend_preflight.py"
    dataset = tmp_path / "images"
    dataset.mkdir()
    for index in range(8):
        (dataset / f"frame_{index:03d}.jpg").write_bytes(b"fixture")
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    out = tmp_path / "out"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--static-only",
            "--expected-head",
            head,
            "--dataset",
            str(dataset),
            "--output-dir",
            str(out),
        ],
        cwd=repo,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert payload["pass"] is True
    assert payload["production_code_changed_by_8_9f1"] is False
    assert payload["external_checkouts_modified_by_8_9f1"] is False
    assert payload["heavy_training_or_inference_started"] is False
    assert payload["source_contracts"]["da3_accepts_external_sfm_poses"] is False
    assert payload["source_contracts"]["current_real_f4_script_uses_deterministic_fixture"] is True
    assert payload["authority"]["gaussian"] == "appearance_only"
    assert payload["authority"]["nerf"] == "appearance_only"
    assert payload["authority"]["certified_metrology"] is False
