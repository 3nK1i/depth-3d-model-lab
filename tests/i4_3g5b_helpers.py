from __future__ import annotations

from core.reconstruction.cross_provider_evidence_synthesis import (
    G5AEvidenceBinding,
    build_i4_3g5a_cross_provider_synthesis,
)
from tests.i4_3g5a_helpers import passing_g1, passing_g2, passing_g3, passing_g4


def passing_g5a() -> dict:
    return build_i4_3g5a_cross_provider_synthesis(
        i4_3g1_snapshot=passing_g1(),
        i4_3g2_closure=passing_g2(),
        i4_3g3c_closure=passing_g3(),
        i4_3g4c_closure=passing_g4(),
        evidence_bindings={
            "i4_3g1_snapshot": G5AEvidenceBinding("/evidence/g1.json", "1" * 64),
            "i4_3g2_closure": G5AEvidenceBinding("/evidence/g2.json", "2" * 64),
            "i4_3g3c_closure": G5AEvidenceBinding("/evidence/g3.json", "3" * 64),
            "i4_3g4c_closure": G5AEvidenceBinding("/evidence/g4.json", "4" * 64),
        },
    )


def passing_g5b_bindings():
    from core.reconstruction.cross_provider_qualification_closure import G5BClosureBinding
    keys = (
        "i4_3g1_snapshot", "i4_3g2_closure", "i4_3g3c_closure",
        "i4_3g4c_closure", "i4_3g5a_synthesis",
    )
    return {
        key: G5BClosureBinding(path=f"/evidence/{key}.json", sha256=f"{idx:064x}")
        for idx, key in enumerate(keys, start=1)
    }
