from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_d_production_adaptation_remains_capability_driven() -> None:
    profile_text = (ROOT / "core" / "hardware" / "profiles.py").read_text(encoding="utf-8").lower()
    reconstruction_text = (ROOT / "core" / "reconstruction" / "profiles.py").read_text(encoding="utf-8").lower()
    # Documentation may mention example GPUs, but decision functions must be driven by memory/capability.
    compute_body = profile_text[profile_text.index("def recommended_profile("):profile_text.index("def resolve_profile(")]
    reconstruction_body = reconstruction_text[
        reconstruction_text.index("def recommended_reconstruction_profile("):
        reconstruction_text.index("def resolve_reconstruction_profile(")
    ]
    for forbidden in ("rtx 2060", "rtx2060", "rtx 4090", "rtx4090", "rtx 5090", "rtx5090"):
        assert forbidden not in compute_body
        assert forbidden not in reconstruction_body
    assert "total_memory_gib" in compute_body
    assert "free_memory_gib" in compute_body
    assert "total_memory_gib" in reconstruction_body
    assert "free_memory_gib" in reconstruction_body
