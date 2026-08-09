from pathlib import Path

from scripts.gaussian_d3_finalization import _isolation_audit


def test_isolation_audit_is_deterministic_across_repeated_calls() -> None:
    root = Path(__file__).resolve().parents[2]
    first = _isolation_audit(root)
    for _ in range(20):
        assert _isolation_audit(root) == first
    assert first["pass"] is True
