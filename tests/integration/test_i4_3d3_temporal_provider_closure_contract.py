import json
import subprocess
import sys
from pathlib import Path

from tests.i4_3d3_helpers import d1_snapshot, d2_snapshot


def test_i43d3_cli_writes_immutable_closure_snapshot(tmp_path: Path) -> None:
    d1 = tmp_path / "d1.json"
    d2 = tmp_path / "d2.json"
    out = tmp_path / "closure"
    d1.write_text(json.dumps(d1_snapshot()), encoding="utf-8")
    d2.write_text(json.dumps(d2_snapshot()), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "scripts/run_i4_3d3_temporal_closure.py", "--i4-3d1-snapshot", str(d1), "--i4-3d2-snapshot", str(d2), "--output-dir", str(out)],
        check=True, capture_output=True, text=True,
    )
    assert "I4.3D3 temporal-provider architecture closure: PASS" in completed.stdout
    payload = json.loads((out / "i4_3d3_temporal_provider_closure.json").read_text(encoding="utf-8"))
    assert payload["decision"]["i4_3e_authorized"] is True
    assert payload["decision"]["provider_promoted"] is False
    assert payload["qualification"]["i4_3g_temporal_video_ab_required"] is True
